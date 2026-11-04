"""Reusable pure-Decode inputs. Prefill/mixed execution retains its existing path."""
import torch


class DecodeInputBuffers:
    def __init__(self, max_seqs, max_model_len, block_size, device="cuda"):
        self.capacity = max_seqs
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.max_blocks = (max_model_len + block_size - 1) // block_size
        self.device = torch.device(device)
        specs = dict(input_ids=torch.int64, positions=torch.int64, slot_mapping=torch.int32,
                     context_lens=torch.int32, temperatures=torch.float32, block_tables=torch.int32)
        # One aligned byte slab keeps every typed field contiguous while reducing
        # six small H2D launches to one. CUDA Graph captures these same GPU views.
        layout, offset = {}, 0
        for name, dtype in specs.items():
            shape = (max_seqs, self.max_blocks) if name == "block_tables" else (max_seqs,)
            count = max_seqs * self.max_blocks if name == "block_tables" else max_seqs
            size = torch.empty((), dtype=dtype, device="cpu").element_size()
            offset = (offset + 7) // 8 * 8
            layout[name] = (offset, count * size, dtype, shape)
            offset += count * size
        self.host_storage = torch.zeros(offset, dtype=torch.uint8, device="cpu",
                                        pin_memory=self.device.type == "cuda")
        self.gpu_storage = torch.zeros(offset, dtype=torch.uint8, device=self.device)
        self.host, self.gpu, self.arrays = {}, {}, {}
        for name, (offset, size, dtype, shape) in layout.items():
            self.host[name] = self.host_storage[offset:offset+size].view(dtype).view(shape)
            self.gpu[name] = self.gpu_storage[offset:offset+size].view(dtype).view(shape)
            self.arrays[name] = self.host[name].numpy()
        self.pending_stream = None

    def fill(self, seqs):
        # Normal execution releases this guard after sampler.to(list) completes.
        # If the preceding run raised, wait before overwriting an in-flight H2D source.
        if self.pending_stream is not None:
            self.pending_stream.synchronize()
            self.pending_stream = None
        n = len(seqs)
        if not 0 < n <= self.capacity:
            raise ValueError("Decode batch exceeds buffer capacity or is empty")
        width = max(len(s.block_table) for s in seqs)
        if width > self.max_blocks or width == 0:
            raise ValueError("Decode block table exceeds buffer capacity or is empty")
        a = self.arrays
        a["slot_mapping"][n:].fill(-1)
        a["context_lens"][n:].fill(0)
        a["block_tables"][:n, :width].fill(-1)
        for i, seq in enumerate(seqs):
            if not 0 < len(seq) <= self.max_model_len:
                raise ValueError("Decode context exceeds configured model length")
            if len(seq.block_table) != (len(seq) + self.block_size - 1) // self.block_size:
                raise ValueError("Decode block table does not cover the context")
            a["input_ids"][i] = seq.last_token
            a["positions"][i] = len(seq)-1
            a["context_lens"][i] = len(seq)
            a["slot_mapping"][i] = seq.block_table[-1]*self.block_size + (len(seq)-1) % self.block_size
            a["temperatures"][i] = seq.temperature
            a["block_tables"][i, :len(seq.block_table)] = seq.block_table
        return n, width

    @torch.inference_mode()
    def upload(self, n, width):
        if self.device.type == "cuda":
            # Mark before the copy, so exceptions cannot permit an unsafe refill.
            self.pending_stream = torch.cuda.current_stream(self.device)
        self.gpu_storage.copy_(self.host_storage, non_blocking=True)
        return {**{name: self.gpu[name][:n] for name in ("input_ids", "positions", "slot_mapping", "context_lens")},
                "block_tables": self.gpu["block_tables"][:n, :width],
                "temperatures": self.gpu["temperatures"][:n]}

    def mark_consumed(self):
        """Only call after the sample result has synchronously reached the CPU."""
        self.pending_stream = None

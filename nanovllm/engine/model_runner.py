import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.registry import get_model_class
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.engine.input_buffers import DecodeInputBuffers
from nanovllm.engine.profiling import StageProfiler


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.profiler = StageProfiler()
        self.decode_buffers = None
        self._direct_decode_graph = False
        self._decode_temperatures = None
        self.dtype = (getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", None)
                      if config.dtype == "auto" else config.dtype)
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype, None)
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported model dtype: {self.dtype}")

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        try:
            torch.cuda.set_device(rank)
            torch.set_default_dtype(self.dtype)
            torch.set_default_device("cuda")
            self.model = get_model_class(hf_config)(hf_config)
            load_model(self.model, config.model)
            for module in self.model.modules():
                if hasattr(module, "batch_invariant"):
                    module.batch_invariant = config.batch_invariant
            self.sampler = Sampler()
            self.decode_buffers = DecodeInputBuffers(config.max_num_seqs, config.max_model_len, self.block_size)
            # Allocate for both A/B variants before sizing KV, keeping resources equal.
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()
        except BaseException:
            reset_context()
            dist.destroy_process_group()
            raise
        finally:
            torch.set_default_device(default_device)
            torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = (getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if config.kv_cache_block_cap is not None:
            config.num_kvcache_blocks = min(config.num_kvcache_blocks, config.kv_cache_block_cap)
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        if self.config.reuse_decode_buffers:
            n, width = self.decode_buffers.fill(seqs)
            self._direct_decode_graph = not self.enforce_eager and n <= 512
            tensors = self.decode_buffers.upload(n, width)
            self._decode_temperatures = tensors["temperatures"]
            set_context(False, slot_mapping=tensors["slot_mapping"], context_lens=tensors["context_lens"],
                        block_tables=tensors["block_tables"])
            return tensors["input_ids"], tensors["positions"]
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        if self._decode_temperatures is not None:
            assert len(seqs) == self._decode_temperatures.numel()
            return self._decode_temperatures
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            if not self._direct_decode_graph:
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"].zero_()
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self._direct_decode_graph = False
        self._decode_temperatures = None
        trace = self.profiler.begin(is_prefill=is_prefill, batch_size=len(seqs),
                    reuse=self.config.reuse_decode_buffers) if self.config.profile_stages else None
        try:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            sample_rows = [i for i, s in enumerate(seqs)
                           if s.num_cached_tokens + s.num_scheduled_tokens == s.num_tokens]
            if is_prefill:
                ends = []
                offset = 0
                for s in seqs:
                    offset += s.num_scheduled_tokens
                    ends.append(offset - 1)
                get_context().sample_indices = torch.tensor(
                    [ends[i] for i in sample_rows], dtype=torch.long, device="cuda")
            temperatures = self.prepare_sample([seqs[i] for i in sample_rows]) if self.rank == 0 else None
            if trace:
                trace.mark("prepare")
            logits = self.run_model(input_ids, positions, is_prefill)
            if trace:
                trace.mark("model")
            if self.rank != 0:
                return None
            sampled = self.sampler(logits, temperatures) if sample_rows else None
            if trace:
                trace.mark("sample")
            values = sampled.tolist() if sampled is not None else []
            if sampled is not None and self.decode_buffers is not None:
                self.decode_buffers.mark_consumed()
            if trace:
                trace.mark("result_wait")
                self.last_profile_trace = trace
                self.profiler.submit(trace)
            result = [None] * len(seqs)
            for i, token in zip(sample_rows, values):
                result[i] = token
            return result
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        # Both baseline and reuse graphs share this layout. Only the way the
        # fields are populated per step differs between the A/B variants.
        buffers = self.decode_buffers.gpu
        self.decode_buffers.gpu_storage.zero_()
        input_ids = buffers["input_ids"][:max_bs]
        positions = buffers["positions"][:max_bs]
        slot_mapping = buffers["slot_mapping"][:max_bs]
        context_lens = buffers["context_lens"][:max_bs]
        block_tables = buffers["block_tables"][:max_bs]
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = sorted({x for x in [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) + [max_bs]
                                if x <= max_bs})
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

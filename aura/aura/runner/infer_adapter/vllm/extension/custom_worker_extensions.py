#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import time
import gc
import json
from acl.rt import memcpy
from typing import List, Optional, Callable, Dict, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.distributed as dist
from safetensors.torch import safe_open
from vllm.distributed import get_world_group

from aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat import vllm_output_statics

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

SLICE_MB = 64
NUM_STREAMS = 8
PREFETCH_DEPTH = 4
SMALL_MAX_BYTES = 8 * (1 << 20)
ACL_MEMCPY_HOST_TO_DEVICE = 1


def calculate_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (float, float):
    """
    Calculate the cosine similarity between two tensors.
    """
    if tensor1.shape != tensor2.shape:
        print(f"Warning: Tensor shape mismatch！{tensor1.shape} vs {tensor2.shape}")
        return 0.0

    vec1 = tensor1.to('cpu').flatten().float()

    vec2 = tensor2.to('cpu', non_blocking=False).flatten().float()

    dot_product = torch.dot(vec1, vec2)
    norm_a = torch.linalg.norm(vec1)
    norm_b = torch.linalg.norm(vec2)

    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if torch.equal(tensor1, tensor2) else 0.0

    similarity = dot_product / (norm_a * norm_b)
    abs_diff = (vec1 - vec2).abs().max()
    return similarity.item(), abs_diff


def _ensure_dtype_config(ts: List[torch.Tensor], dtype: torch.dtype) -> List[torch.Tensor]:
    out = []
    for t in ts:
        if t.dtype is not dtype:
            t = t.to(dtype, copy=True)
        if not t.is_contiguous():
            t = t.contiguous()
        out.append(t)
    return out


def _bytes_of(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _list_rank_files(final_dir: str, r: int) -> List[str]:
    prefix = f"rank_{r}_"
    files = []
    with os.scandir(final_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if not name.startswith(prefix):
                continue
            if not name.endswith(".safetensors"):
                continue
            files.append(entry.path)
    files.sort()
    return files


def iter_rank_tensors(
    files: Optional[List[str]] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Lazily iterate over (key, tensor) from all .safetensors files in `files`
    whose filename starts with f"rank_{r}".

    - Uses memory mapping via safetensors, opening one file at a time.
    - Yields tensors one-by-one to keep RSS low.
    - Optionally casts floating-point tensors to `dtype`.
    - Optionally filters keys via `key_filter(key) -> bool`.

    Args:
        files: lists of shard files.
        device: Target device for loaded tensors ("cpu", "cuda", "npu", etc.).
        dtype: Optional target dtype for *floating-point* tensors only.
        key_filter: Optional predicate to select keys to load.

    Yields:
        (key, tensor) for each tensor found across matching files.
    """
    files.sort()

    for path in files:
        # safe_open is lazy + mmapped; device controls where get_tensor loads to
        with safe_open(path, framework="pt", device=device) as f:
            keys = f.keys()
            if key_filter is not None:
                keys = [k for k in keys if key_filter(k)]
            for k in keys:
                t = f.get_tensor(k)  # loaded on-demand, on chosen device
                if dtype is not None and t.is_floating_point() and t.dtype != dtype:
                    t = t.to(dtype)
                yield k, t


def load_rank_tensors_to_dict(
    files: Optional[List[str]] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    key_filter: Optional[Callable[[str], bool]] = None,
    on_duplicate: str = "overwrite",
) -> dict[str, torch.Tensor]:
    """
    Convenience wrapper that builds a dict. For very large models prefer the iterator.

    on_duplicate: "overwrite" or "keep_first"
    """
    out: dict[str, torch.Tensor] = {}
    for k, t in iter_rank_tensors(files, device=device, dtype=dtype, key_filter=key_filter):
        if k in out and on_duplicate == "keep_first":
            continue
        out[k] = t
    return out


def check_similarity_enable():
    enable_str = os.environ.get("ENABLE_TENSOR_SIMILARITY_CHECK", 'false').lower()
    return enable_str == 'true'


def print_similarities(keys, cpu_list, npu_list, addition_info=""):
    if not check_similarity_enable():
        return
    print(f"\n--- Tensor {addition_info} Cosine Similarity Check Results ---")
    for i in range(len(keys)):
        similarity, abs_diff = calculate_cosine_similarity(cpu_list[i], npu_list[i])
        print(f"🔑 Key: {keys[i]:20} | {addition_info} Cosine Similarity: {similarity:.6f} abs_diff: {abs_diff:.6f}")
    print("--------------------------------------------\n")


@torch.no_grad()
def _core_transfer_to_npu(
    cpu_tensors: Dict[str, torch.Tensor],
    *,
    target_dtype: Optional[torch.dtype] = torch.bfloat16,
    npu_targets: Optional[Dict[str, torch.Tensor]] = None,
    device: str | torch.device = "npu:0",
    num_streams: int = NUM_STREAMS,
    use_memcpy: bool = False,
) -> Dict[str, torch.Tensor]:
    if not all(t.device.type == "cpu" for t in cpu_tensors.values()):
        raise ValueError("All tensors in cpu_tensors must be on CPU device")
    keys = list(cpu_tensors.keys())
    cpu_list = list(cpu_tensors.values())

    if target_dtype is None:
        target_dtype = next(iter(cpu_tensors.values())).dtype

    cpu_list = _ensure_dtype_config(cpu_list, target_dtype)

    if npu_targets is None:
        npu_list = [torch.empty_like(t, device=device) for t in cpu_list]
    else:
        if set(npu_targets.keys()) != set(cpu_tensors.keys()):
            raise ValueError(
                f"npu_targets keys {set(npu_targets.keys())} do not match cpu_tensors keys {set(cpu_tensors.keys())}"
            )
        npu_list = [npu_targets[k] for k in keys]
        for i in range(len(cpu_list)):
            s = cpu_list[i]
            d = npu_list[i]
            if s.shape != d.shape:
                logger.info(f"key: {keys[i]}, cpu shape: {s.shape}, npu shape: {d.shape}")
                if len(s.shape) == 3 and s.shape[1] == d.shape[2]:
                    s = s.permute(0, 2, 1)
                    cpu_list[i] = s
            if d.device.type != "npu" or d.shape != s.shape or d.dtype != target_dtype:
                raise ValueError(
                    f"Tensor device/shape/dtype mismatch: device={d.device.type} (expected npu), "
                    f"shape={d.shape} (expected {s.shape}), dtype={d.dtype} (expected {target_dtype})"
                )

    dev = torch.device(device)
    torch.npu.set_device(dev)

    # Use with torch.npu.device(dev) to activate the NPU context
    with torch.npu.device(dev):
        print_similarities(keys, cpu_list, npu_list, "PreCheck")

        streams = [torch.npu.Stream(device=dev) for _ in range(num_streams)]
        si = 0

        for i in range(len(cpu_list)):
            sm = streams[si]
            with torch.npu.stream(sm):
                _direct_copy(cpu_list[i], npu_list[i], use_memcpy)
            si = (si + 1) % num_streams

        torch.npu.synchronize(dev)
        print_similarities(keys, cpu_list, npu_list, "ReCheck")

    return {k: n for k, n in zip(keys, npu_list)}


def _direct_copy(src, dst, use_memcp=False):
    if src.shape != dst.shape:
        raise ValueError(f"src shape {src.shape} does not match dst shape {dst.shape}")
    if use_memcp:
        n = src.numel() * src.element_size()
        cpu_ptr = src.data_ptr()
        npu_ptr = dst.data_ptr()
        memcpy(npu_ptr, n, cpu_ptr, n, ACL_MEMCPY_HOST_TO_DEVICE)
    else:
        with torch.no_grad():
            dst.copy_(src, non_blocking=True)


def _thread_load_and_h2d(
    file_chunk: List[str],
    npu_targets: Optional[Dict[str, torch.Tensor]],
    *,
    target_dtype: Optional[torch.dtype],
    device: str,
    num_streams: int,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> Dict[str, object]:
    t0 = time.time()
    cpu_dict = load_rank_tensors_to_dict(
        file_chunk,
        device="cpu",
        dtype=target_dtype,  # cast once on CPU
        key_filter=key_filter,
        on_duplicate="overwrite",
    )

    t1 = time.time()
    npu_targets = {k: npu_targets[k] for k in cpu_dict.keys()} if (npu_targets is not None) else None
    _ = _core_transfer_to_npu(
        cpu_dict,
        target_dtype=target_dtype,
        npu_targets=npu_targets,
        device=device,
        num_streams=num_streams,
        use_memcpy=False,
    )

    t2 = time.time()

    num_tensors = len(cpu_dict)
    num_bytes = sum(_bytes_of(t) for t in cpu_dict.values())
    cpu_dict.clear()
    gc.collect()

    state_dict = {
        "num_files": len(file_chunk),
        "num_tensors": num_tensors,
        "bytes_cpu": num_bytes,
        "t_cpu_load_s": t1 - t0,
        "t_h2d_s": t2 - t1,
        "t_total_s": t2 - t0,
    }
    logger.info(f"h2d state_dict: {state_dict}")
    return state_dict


def load_rank_to_npu_threaded(
    final_dir: str,
    r: int,
    num_threads: int,
    npu_targets: Optional[Dict[str, torch.Tensor]] = None,
    *,
    target_dtype: Optional[torch.dtype] = torch.bfloat16,
    device: str = "npu:0",
    num_streams: int = NUM_STREAMS,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> Dict[str, object]:
    files = _list_rank_files(final_dir, r)
    logger.info(f"*********rank={r} final_dir={final_dir} files={files}*********")
    if not files:
        logger.info("*********not file********")
        return {"threads": 0, "num_files": 0, "bytes_total": 0, "by_thread": [], "note": "no files"}

    threads = max(1, min(num_threads, len(files)))
    per = (len(files) + threads - 1) // threads
    chunks = [files[i * per : (i + 1) * per] for i in range(threads) if files[i * per : (i + 1) * per]]

    by_thr = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        results = [
            ex.submit(
                _thread_load_and_h2d,
                chunk,
                npu_targets,
                target_dtype=target_dtype,
                device=device,
                num_streams=num_streams,
                key_filter=key_filter,
            )
            for chunk in chunks
        ]
        for f in as_completed(results):
            by_thr.append(f.result())
    total_bytes = sum(t["bytes_cpu"] for t in by_thr)

    return {
        "threads": len(chunks),
        "num_files": len(files),
        "bytes_total": total_bytes,
        "by_thread": by_thr,
        "note": "Loaded on threads; H2D wrote into provided npu_targets if given.",
    }


def _current_accel_device():
    if hasattr(torch, "npu"):
        return torch.device("npu", torch.npu.current_device())
    elif torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    raise RuntimeError("No NPU/CUDA backend found.")


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    # Prefer Ascend NPUs if present
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    # Then CUDA
    if torch.cuda.is_available():
        return torch.device("cuda", index=torch.cuda.current_device())
    # Fallback CPU
    return torch.device("cpu")


def broadcast_if_gpu(tensor, src, group):
    """Broadcast only when the tensor resides on an NPU/GPU."""
    if tensor.device.type in ("npu", "cuda"):
        dist.broadcast(tensor, src=src, group=group, async_op=False)


def split_tensors_and_meta(params: dict):
    """Return (tensor_params, meta_str_dict). meta_str_dict is safe for safetensors header."""
    params = dict(params)  # shallow copy; we’ll pop
    meta = params.pop("__simple_ep_meta__", None)

    # Keep only tensors; drop anything else defensively
    tensor_params = {k: v for k, v in params.items() if isinstance(v, torch.Tensor)}

    # Pack meta as a single JSON string under a reserved key
    meta_header = {}
    if meta is not None:
        meta_header["__simple_ep_meta__"] = json.dumps(meta)
    return tensor_params, meta_header


def _vocab_size_check(k, cpu_tensor):
    if "model.embed_tokens.weight" in k or "lm_head.weight" in k:
        actual_vocab_size = cpu_tensor.shape[0]
        logger.info(f"core weight check: Key={k}, shape={cpu_tensor.shape}, vocab_size={actual_vocab_size}")


class CustomWorkerExtensions:
    def update_weights_with_assemble(self, dir_path):
        name2params = dict(self.model_runner.model.named_parameters())
        logger.info(f"################### {name2params.keys()=}")
        dev = torch.npu if hasattr(torch, "npu") else torch.cuda
        device = _current_accel_device()
        dtype = next(iter(name2params.values())).dtype

        logger.info(f"dir_path: {dir_path}")
        t0 = time.perf_counter()
        world_rank = torch.distributed.get_rank(group=get_world_group().cpu_group)
        stats = load_rank_to_npu_threaded(
            final_dir=dir_path, r=world_rank, npu_targets=name2params, target_dtype=dtype, device=device, num_threads=8
        )
        dev.synchronize()
        cost = time.perf_counter() - t0
        moved_bytes = stats["bytes_total"]
        logger.info(
            f"|perf-stat|infer-server| rank {world_rank} update_weights took: {cost:.2f}s, moved bytes: {moved_bytes}"
        )
        return moved_bytes

    def update_weights_with_hf(self, dir_path):
        start = time.perf_counter()
        logger.info(f"|perf-stat|train| start updating weights: {self.rank=}, dir_path={dir_path}")
        files = []
        with os.scandir(dir_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.endswith(".safetensors"):
                    continue
                files.append(entry.path)
        files.sort()
        logger.info(f">>>> weight files: {files}")
        if not files:
            logger.warning(">>>> no weights to update")
            return 0

        from verl.utils.device import get_device_name, get_device_id

        device_str = f"{get_device_name()}:{get_device_id()}"
        logger.info(f" >>> {device_str=}")

        transpose_shape = os.getenv("TRANSPOSE_EXPERT_SHAPE", "false")
        logger.info(f" >>> transpose_shape={transpose_shape}")

        def iter_weights():
            for fp in files:
                logger.info(f"start load weights from file: {fp}")
                with safe_open(fp, framework="pt") as f:
                    for k in f.keys():
                        cpu_tensor = f.get_tensor(k)
                        _vocab_size_check(k, cpu_tensor)

                        device_tensor = cpu_tensor.to(device_str, non_blocking=False)
                        if "experts" in k and len(device_tensor.shape) == 3 and transpose_shape == "true":
                            if "down_proj" in k or "up_proj" in k or "gate_proj" in k:
                                device_tensor = device_tensor.transpose(1, 2).contiguous()

                        yield k, device_tensor

                        del cpu_tensor
            import gc

            gc.collect()

        inference_model = self.model_runner.model
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(inference_model)
        inference_model.load_weights(iter_weights())

        cost = time.perf_counter() - start
        logger.info(f"|perf-stat|train| weight update completed: {self.rank=}, cost={cost:.2f}s")
        return 0

    @torch.no_grad()
    def update_weights_with_disk(self, *args, **kwargs):
        dir_path = ''.join(args)
        train_backend = os.getenv("RL_TRAIN_BACKEND", "mindspeed_rl")
        logger.info(f"dir_path: {dir_path}, train_backend: {train_backend}")
        if train_backend == "verl":
            return self.update_weights_with_hf(dir_path)
        return self.update_weights_with_assemble(dir_path)

    def vllm_statistics(self, *args, **kwargs):
        logger.info(f"|perf-stat|infer-server| vllm_statistics begin, {self.rank=}")
        vllm_output_statics.write_stats_tofile()

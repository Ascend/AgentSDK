#!/usr/bin/env python3
# coding=utf-8
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
import glob
import json
import multiprocessing as mp
import os
import re
import shutil
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Iterable, List, Tuple, Optional

import ray
import torch
from ray.exceptions import GetTimeoutError
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from safetensors.torch import safe_open, save_file

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

torch.set_num_threads(16)
torch.set_num_interop_threads(1)

# Filenames must look like: ".../pp0_tp0_ep{N}.safetensors"
_PP_TP_EP_RE = re.compile(r".*pp(\d+).*_tp(\d+)(?:_ep(\d+))?\.safetensors$")
_REMOVE_IDX_RE = re.compile(r"\.\d+\.")


def _norm(name: str) -> str:
    # remove ".{digits}." segments once, deterministically
    return _REMOVE_IDX_RE.sub(".", name)


def cat_dim0(parts: List[torch.Tensor]) -> torch.Tensor:
    if len(parts) == 1: return parts[0]
    h_shape = sum(p.shape[0] for p in parts)
    w_shape = parts[0].shape[1]
    out = torch.empty((h_shape, w_shape), dtype=parts[0].dtype)
    s = 0
    for p in parts:
        e = s + p.shape[0]
        out[s:e, :].copy_(p)  # single pass copy
        s = e
    return out


def cat_dim1(parts: List[torch.Tensor]) -> torch.Tensor:
    if len(parts) == 1: return parts[0]
    h_shape = parts[0].shape[0]
    w_shape = sum(p.shape[1] for p in parts)
    out = torch.empty((h_shape, w_shape), dtype=parts[0].dtype)
    s = 0
    for p in parts:
        e = s + p.shape[1]
        out[:, s:e].copy_(p)
        s = e
    return out


def _fast_cat(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
    if len(tensors) == 0:
        raise ValueError("fast_cat received an empty tensor list")

    if len(tensors) == 1:
        return tensors[0]

    tensors = [t.contiguous() for t in tensors]
    cat = torch.cat(tensors, dim=dim)
    return cat


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _chunk_evenly(items: list, workers: int) -> List[List]:
    if not items:
        return []
    workers = max(1, min(workers, len(items)))
    per = _ceil_div(len(items), workers)
    return [items[i * per:(i + 1) * per] for i in range(workers) if items[i * per:(i + 1) * per]]


@dataclass
class LaunchedGroup:
    refs: list
    pg: Any | None
    flatten: bool = True

    def get(self):
        try:
            out = ray.get(self.refs)
            if self.flatten and out and isinstance(out[0], list):
                flat = []
                for sub in out: flat.extend(sub)
                return flat
            return out
        finally:
            if self.pg is not None:
                try:
                    remove_placement_group(self.pg)
                except Exception as exp:
                    logger.warning(f"remove placement group failed: {exp}")
                    pass
                self.pg = None


def plan_num_cpus(desired_workers: int, ideal_cpus_per_worker: int) -> int:
    max_use_rate = 0.75
    free_cpus = int(ray.available_resources().get("CPU", 0))
    if free_cpus * max_use_rate < desired_workers * ideal_cpus_per_worker:
        return max(1, free_cpus // max(1, 2 * desired_workers))
    return max(1, ideal_cpus_per_worker)


def create_pg_with_fallback(*, workers: int, bundle_cpus: int, timeout_s: float = 5.0,
                            min_cpus: int = 1, strategy: str = "SPREAD"):
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    b = max(min_cpus, bundle_cpus)
    while b >= min_cpus:
        pg = placement_group(bundles=[{"CPU": b}] * workers, strategy=strategy)
        try:
            ray.get(pg.ready(), timeout=timeout_s)
            return pg, b
        except GetTimeoutError:
            remove_placement_group(pg)
            nb = max(min_cpus, b // 2)
            if nb == b:
                break
            b = nb
    raise RuntimeError(f"Could not place PG with ≥{min_cpus} CPU(s) per bundle (workers={workers}).")


def launch_chunked_with_pg(
        *,
        items: list,
        workers: int,
        ideal_cpus_per_worker: int,
        make_kwargs,
        capture_child_tasks: bool = True,
        strategy: str = "SPREAD",
        timeout_s: float = 5.0,
        use_pg_if_one: bool = False, flatten_list_results: bool = True
) -> LaunchedGroup:
    chunks = _chunk_evenly(items, workers)
    if not chunks:
        return LaunchedGroup(refs=[], pg=None, flatten=flatten_list_results)
    if not use_pg_if_one and len(chunks) == 1:
        num_cpus = plan_num_cpus(1, ideal_cpus_per_worker)
        kw = make_kwargs(0, chunks[0])
        ref = assemble_subset_worker.options(num_cpus=num_cpus).remote(**kw)
        return LaunchedGroup(refs=[ref], pg=None, flatten=flatten_list_results)

    bundle_cpus = plan_num_cpus(len(chunks), ideal_cpus_per_worker)
    pg = None
    try:
        pg, final_cpus = create_pg_with_fallback(workers=len(chunks), bundle_cpus=bundle_cpus,
                                                 timeout_s=timeout_s, strategy=strategy)
        refs = []
        for i, chunk in enumerate(chunks):
            strat = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
                placement_group_capture_child_tasks=capture_child_tasks,
            )
            kw = make_kwargs(i, chunk)
            refs.append(assemble_subset_worker.options(scheduling_strategy=strat, num_cpus=final_cpus).remote(**kw))
        return LaunchedGroup(refs=refs, pg=pg, flatten=flatten_list_results)
    except Exception:
        if pg is not None:
            try:
                remove_placement_group(pg)
            except Exception as exp:
                logger.warning(f"remove placement group failed: {exp}")
                pass
        raise


def _parse_pp_tp_ep(path: str) -> Tuple[int, int, int]:
    m = _PP_TP_EP_RE.match(path)
    if not m:
        raise ValueError(f"Cannot parse (pp,tp,ep) from file name: {path}")
    pp, tp, ep = m.groups()
    pp, tp = int(pp), int(tp)
    ep = int(ep) if ep is not None else 0  # default EP=0 when not present
    return pp, tp, ep


class _ReaderCache:
    """Keep a small LRU of safetensors.safe_open handles; thread-safe per file."""

    def __init__(self, max_open: int = 64, copy_on_read: bool = False):
        self.max_open = max_open
        self._cache = {}  # path -> (handle, lock)
        self._order = deque()  # LRU

        self.copy_on_read = copy_on_read

    def get(self, path: str):
        entry = self._cache.get(path)
        if entry is not None:
            # move to the end (most recent)
            try:
                self._order.remove(path)
            except ValueError:
                pass
            self._order.append(path)
            return entry
        f = safe_open(path, framework="pt", device="cpu")
        lock = Lock()
        self._cache[path] = (f, lock)
        self._order.append(path)
        if len(self._order) > self.max_open:
            old = self._order.popleft()
            try:
                self._cache[old][0].close()
            except Exception as exp:
                logger.warning(f"close cache failed: {exp}")
                pass
            del self._cache[old]
        return self._cache[path]

    def close_all(self):
        for p, (f, _) in list(self._cache.items()):
            try:
                f.close()
            except Exception as exp:
                logger.warning(f"close file failed: {exp}")
                pass
        self._cache.clear()
        self._order.clear()

    def load_tensor(self, path: str, k: str) -> torch.Tensor:
        f, lock = self.get(path)
        with lock:
            t = f.get_tensor(k)

        if self.copy_on_read:
            # copy=True guarantees a fresh heap allocation
            return t.to(t.dtype, copy=True).contiguous()
        return t

    def read_meta(self, path: str) -> Optional[dict]:
        f, lock = self.get(path)
        with lock:
            return self._read_meta_locked(f)

    def _read_meta_locked(self, f) -> Optional[dict]:
        """Return parsed __simple_ep_meta__ if present, else None."""
        md = f.metadata()
        if not md:
            return None
        raw = md.get("__simple_ep_meta__")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as exp:
            logger.warning(f"json loads failed: {exp}")
            return None


def list_all_keys(pp2ep_paths, rc) -> List[str]:
    keys = set()
    for pp, ep2tplist in pp2ep_paths.items():
        ep = min(ep2tplist)
        tp = min(tp for tp, _ in ep2tplist[ep])
        path = dict(ep2tplist[ep])[tp]
        f, lock = rc.get(path)
        with lock:
            keys.update(f.keys())
    return sorted(keys)


class ParamsAssembler:
    """
    Simplified assembler for TP=1, PP=1 checkpoints that may have EP sharding.
    target_dtype: optional torch dtype (e.g., torch.bfloat16) to cast outputs.
    """
    target_dtype: Optional[torch.dtype] = None
    infer_tp: int = 1
    infer_dp: int = 1
    w13_gate_up_order: str = "gate_up"
    has_moe: bool = True
    remove_idx_pattern = re.compile(r"\.\d+\.")
    num_attention_heads = None
    num_key_value_heads = None
    head_dim = None
    num_layers = None
    hidden_size = None
    num_experts = None
    w2_rows_mode = "grouped"
    w13_cols_mode = "interleaved"
    head_dim_scale = 1
    use_simple_ep_mode = False

    def __init__(self, infer_tp: int = 1, infer_dp: int = 1, target_dtype: Optional[torch.dtype] = None,
                 head_dim_scale: int = 1, use_simple_ep_mode: bool = False):
        self.target_dtype = target_dtype
        self.infer_tp = max(infer_tp, 1)
        self.infer_dp = max(infer_dp, 1)
        self.head_dim_scale = max(head_dim_scale, 1)
        self.use_simple_ep_mode = use_simple_ep_mode

        logger.info(f"infer_tp: {infer_tp}, infer_dp: {infer_dp}")

    @property
    def _Teff(self) -> int:
        # Interleave factor for layouts that depend on parallelism along "per-expert" dims
        return self.infer_tp * self.infer_dp

    def _cast_if_needed(self, t: torch.Tensor) -> torch.Tensor:
        if self.target_dtype and t.is_floating_point() and t.dtype != self.target_dtype:
            return t.to(self.target_dtype)
        return t

    def get_tp_split_axis(self, name: str) -> Optional[int]:
        return None

    def is_fused_qkv_weight(self, name: str) -> bool:
        return False

    def is_fused_qkv_bias(self, name: str) -> bool:
        return False

    def is_w13(self, name: str) -> bool:
        return False

    def is_w2(self, name: str) -> bool:
        return False

    def get_train_tp_concat_axis_2d(self, name: str) -> int | None:
        return None

    def get_weight_3D_shape(self, name: str, old_shape: List[int]) -> Optional[Tuple]:
        return None

    def get_weight_3D_permute(self, name: str) -> Optional[Tuple]:
        return None

    def unflatten_weight(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        old_shape = list(weight.shape)
        if len(old_shape) == 3:
            return weight
        new_shape = self.get_weight_3D_shape(name, old_shape)
        perm = self.get_weight_3D_permute(name)
        if (new_shape is None) or (perm is None):
            return weight
        return weight.view(*new_shape).permute(*perm).contiguous()

    def _repack_w13_flat_fused(self, t2d: torch.Tensor, Ewin: int, name: str) -> torch.Tensor:
        """
        Bit-exact fused repack+collapse for w13:
        Input  (per EP window, TP-major cat): t2d [H, Ewin*U]
        Output (flattened, post-collapse):    [H, Ewin*U]
        It reproduces the old path:
        group-by-expert -> view(H,E,2,T,c) -> permute(..., c, T) -> collapse T (innermost)
        """
        if t2d is None or t2d.dim() != 2:
            raise ValueError(f"{name}: expected 2D tensor, got {None if t2d is None else tuple(t2d.shape)}")

        H, Wwin = t2d.shape
        if Wwin % Ewin != 0:
            raise ValueError(f"{name}: window width {Wwin} not divisible by Ewin={Ewin}")

        T = int(self._Teff)
        U = Wwin // Ewin
        if U % (2 * T) != 0:
            raise ValueError(f"{name}: U={U} must be divisible by 2*TP_eff={2 * T}")
        c = U // (2 * T)

        # Exact expert-grouping reorder like the old method:
        # old did: t2d.view(Ewin, H, U).permute(1, 0, 2).contiguous() -> [H, Ewin, U]
        g = t2d.view(Ewin, H, U).permute(1, 0, 2)  # [H, Ewin, U] (no contiguous yet)
        flat = (g.reshape(H, Ewin, 2, T, c)  # reshape tolerates non-contig (copies once)
                .permute(0, 1, 2, 4, 3)
                .reshape(H, Ewin * U))
        return flat

    def reshape_qkv_megatron_local(self, name, query_key_value):
        if self.is_fused_qkv_weight(name):
            nh = self.num_attention_heads // self.infer_tp
            ng = self.num_key_value_heads // self.infer_tp
            repeats = nh // ng
            head_dim = query_key_value.shape[0] // (ng * (repeats + 2))
            if (head_dim * self.head_dim_scale) != self.head_dim:
                raise ValueError(f"weight head_dim mismatch: {head_dim * self.head_dim_scale} != {self.head_dim}")
            qkv_weight = query_key_value.reshape(ng, repeats + 2, head_dim, query_key_value.shape[1])
            hidden_size = qkv_weight.shape[-1]
            qw = qkv_weight[:, :repeats, ...].reshape(-1, hidden_size)
            kw = qkv_weight[:, repeats: repeats + 1, ...].reshape(-1, hidden_size)
            vw = qkv_weight[:, repeats + 1:, ...].reshape(-1, hidden_size)
            return torch.cat([qw, kw, vw], dim=0)
        if self.is_fused_qkv_bias(name):
            nh = self.num_attention_heads // self.infer_tp
            ng = self.num_key_value_heads // self.infer_tp
            repeats = nh // ng
            head_dim = query_key_value.shape[0] // (ng * (repeats + 2))
            if (head_dim * self.head_dim_scale) != self.head_dim:
                raise ValueError(f"weight head_dim mismatch: {head_dim * self.head_dim_scale} != {self.head_dim}")
            bias_weight = query_key_value.reshape(
                ng,
                repeats + 2,
                head_dim
            )
            qw = bias_weight[:, :repeats, ...].reshape(-1)
            kw = bias_weight[:, repeats: repeats + 1, ...].reshape(-1)
            vw = bias_weight[:, repeats + 1:, ...].reshape(-1)
            return torch.cat([qw, kw, vw], dim=0)

        return query_key_value

    # ---------- I/O helpers ----------
    @staticmethod
    def group_paths_by_pp_ep_tp(paths: Iterable[str]) -> Dict[int, Dict[int, List[Tuple[int, str]]]]:
        """
        Choose ONE file per (pp, ep, tp) group.
        """
        buckets = {}
        for p in paths:
            pp, tp, ep = _parse_pp_tp_ep(p)
            buckets.setdefault(pp, {}).setdefault(ep, []).append((tp, p))
        for pp in buckets:
            for ep in buckets[pp]:
                buckets[pp][ep].sort(key=lambda x: x[0])  # by tp
        return buckets

    # ---------- core assembly ----------
    def assemble_dir(self, pp2ep_paths: Dict[int, Dict[int, List[Tuple[int, str]]]], rc: _ReaderCache,
                     names_filter: Optional[set] = None) -> Dict[str, torch.Tensor]:
        final_tensors: Dict[str, torch.Tensor] = {}
        error = None
        # logger.info("Started assemble loop.")
        try:
            for pp, ep2tplist in sorted(pp2ep_paths.items()):
                # ep -> {tp: path}
                ep_tp_path = {ep: {tp: path for (tp, path) in tplist}
                              for ep, tplist in ep2tplist.items()}
                all_eps = sorted(ep_tp_path.keys())
                all_tps = sorted({tp for ep in all_eps for tp in ep_tp_path[ep]})

                if not all_tps:
                    continue

                # Read per-EP metadata (from any TP file)
                metas: Dict[int, dict] = {}
                for ep in all_eps:
                    tp0 = min(ep_tp_path[ep].keys())
                    metas[ep] = rc.read_meta(ep_tp_path[ep][tp0]) or {}

                # pick a reference (ep, tp) to list keys
                ref_ep = 0 if 0 in ep_tp_path else min(all_eps)
                ref_tp = min(ep_tp_path[ref_ep].keys())
                f_ref, lock = rc.get(ep_tp_path[ref_ep][ref_tp])
                with lock:
                    ref_keys = set(f_ref.keys())

                # build per-(ep,tp) metadata
                pair_meta = {}
                for ep in all_eps:
                    for tp, path in ep_tp_path[ep].items():
                        pair_meta[(ep, tp)] = rc.read_meta(path) or {}

                # collect expert keys from all pairs
                expert_keys = set()
                for md in pair_meta.values():
                    expert_keys |= set(md.get("slices", {}).keys())

                non_expert_keys = ref_keys - expert_keys

                if names_filter is not None:
                    expert_keys = {k for k in expert_keys if k in names_filter}
                    non_expert_keys = {k for k in non_expert_keys if k in names_filter}

                # sanity/meta flags
                meta0 = metas.get(ref_ep) or {}
                num_experts_total = meta0.get("num_experts_total")
                if num_experts_total is None:
                    num_experts_total = self.num_experts

                # Build once per (pp group), right after pair_meta is built
                key2entries = defaultdict(list)
                for ep in all_eps:
                    for tp, path in ep_tp_path[ep].items():
                        sl = pair_meta[(ep, tp)].get("slices", None)
                        if not sl:
                            continue
                        for k, v in sl.items():
                            key2entries[k].append((
                                ep, tp, path,
                                int(v["offset"]), int(v["length"]), int(v.get("axis", 1)),
                            ))

                # ---------------- Experts: stitch EP per TP -> cat across TP -> 3D ----------------
                with torch.no_grad():
                    for key in sorted(expert_keys):
                        # print(f"Cur key: {key}")
                        # locate an EP that has slices meta for this key
                        some_ep = next((ep for ep in all_eps if key in (metas[ep].get("slices", {}))), None)
                        if some_ep is None:
                            continue
                        sl_meta = metas[some_ep]["slices"][key]
                        expert_axis = int(sl_meta.get("axis", 0))  # 0 for w2 rows, 1 for w13 cols

                        # --- Build 2-D by stitching across EP, but pick exactly ONE TP file
                        seed_ep, seed_tp = next((p for p in pair_meta if key in pair_meta[p].get("slices", {})))
                        seed_part = rc.load_tensor(ep_tp_path[seed_ep][seed_tp], key)
                        sl0 = pair_meta[(seed_ep, seed_tp)]["slices"][key]
                        E_local = int(sl0["length"])
                        expert_axis = int(sl0["axis"])

                        # Build buckets[(ep, o, ln)] -> list[(tp, tensor)]
                        entries = key2entries.get(key, ())
                        if not entries:
                            continue

                        # Group per window (ep,o,ln)
                        buckets = defaultdict(list)
                        for ep, tp, path, o, ln, _axis in entries:
                            buckets[(ep, o, ln)].append((tp, path))

                        # How many TP shards contribute to ONE expert window?
                        any_bucket = next(iter(buckets.values()))
                        tp_in_window = len(any_bucket)

                        if expert_axis == 0:  # w2: (E_local*per_tp, H) per shard
                            per_tp = seed_part.shape[0] // E_local
                            per_total = per_tp * tp_in_window
                            # FINAL full matrix must have per_total per expert
                            full2d = torch.empty((num_experts_total * per_total, self.hidden_size),
                                                 dtype=seed_part.dtype, device="cpu")
                            # Buffer sized for max window: ln ≤ E_local -> E_local * per_total rows
                            cat_buf = torch.empty(E_local * per_total, self.hidden_size,
                                                  dtype=seed_part.dtype, device="cpu")
                        else:  # w13: (H, E_local*per_tp) per shard
                            per_tp = seed_part.shape[1] // E_local
                            per_total = per_tp * tp_in_window  # this is U (cols/expert before gate/up split)
                            # FINAL full matrix must have per_total per expert (already TP-collapsed via fused repack)
                            full2d = torch.empty((self.hidden_size, num_experts_total * per_total),
                                                 dtype=seed_part.dtype, device="cpu")
                            # Buffer sized for max window: ln ≤ E_local -> H x (E_local * per_total) cols
                            cat_buf = torch.empty(self.hidden_size, E_local * per_total,
                                                  dtype=seed_part.dtype, device="cpu")

                        is_w13_key = self.is_w13(key)
                        is_w2_key = self.is_w2(key)

                        # ----- Per-window copy using cat_buf (fused) -----
                        for (ep, o, ln), lst in buckets.items():
                            off = 0
                            if len(lst) == 1:
                                tp, path = lst[0]
                                t = rc.load_tensor(ep_tp_path[ep][tp], key)

                                if expert_axis == 0:
                                    cat = self._w2_interleave_rows(t, ln) if is_w2_key else t
                                    per_here = cat.shape[0] // ln
                                    s, e = o * per_here, (o + ln) * per_here
                                    full2d[s:e, :].copy_(cat)
                                else:
                                    cat_in = t
                                    cat = self._repack_w13_flat_fused(cat_in, Ewin=ln,
                                                                      name=key) if is_w13_key else cat_in
                                    per_here = cat.shape[1] // ln
                                    s, e = o * per_here, (o + ln) * per_here
                                    full2d[:, s:e].copy_(cat)
                                continue
                            if expert_axis == 0:
                                # row-wise concat across training TP
                                for _, t in lst:
                                    h = ln * per_tp  # rows contributed by this TP shard
                                    cat_buf[off:off + h, :].copy_(t)
                                    off += h
                                cat_win = cat_buf[:off, :]  # [ln * per_total, H]
                                cat = self._w2_interleave_rows(cat_win, ln) if self.is_w2(key) else cat_win

                                per_here = cat.shape[0] // ln  # == per_total
                                s, e = o * per_here, (o + ln) * per_here
                                full2d[s:e, :].copy_(cat)

                            else:
                                # col-wise concat across training TP
                                for _, t in lst:
                                    w = ln * per_tp  # cols contributed by this TP shard
                                    cat_buf[:, off:off + w].copy_(t)
                                    off += w
                                cat_win = cat_buf[:, :off]  # [H, ln * per_total]

                                if self.is_w13(key):
                                    # fused repack + TP-collapse → bit-exact with old repack+collapse
                                    cat = self._repack_w13_flat_fused(cat_win, Ewin=ln, name=key)  # [H, ln * per_total]
                                else:
                                    cat = cat_win  # non-w13 axis==1: just the filled slice

                                per_here = cat.shape[1] // ln  # == per_total
                                s, e = o * per_here, (o + ln) * per_here
                                full2d[:, s:e].copy_(cat)

                        final3d = self.unflatten_weight(key, full2d)
                        final_tensors[key] = self._cast_if_needed(final3d).contiguous()

                # ---------------- Non-experts: cat TP for a single EP (identical across EP) ----------------
                ref_ep_for_non_expert = ref_ep
                with torch.no_grad():
                    for key in sorted(non_expert_keys):
                        parts_tp = []
                        for tp in sorted(ep_tp_path[ref_ep_for_non_expert].keys()):
                            parts_tp.append(rc.load_tensor(ep_tp_path[ref_ep_for_non_expert][tp], key))

                        tp_cat_axis = self.get_train_tp_concat_axis_2d(key)
                        if tp_cat_axis is None:
                            final2d = parts_tp[0]  # not sharded by TP
                        else:
                            final2d = _fast_cat(parts_tp, dim=tp_cat_axis)

                        final_tensors[key] = self._cast_if_needed(final2d).contiguous()

            return final_tensors
        except Exception as e:
            error = e
        finally:
            rc.close_all()
            if error is not None:
                raise ValueError(error)

    def write_file(self, tensors: Dict[str, torch.Tensor], out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        save_file({k: v.contiguous() for k, v in tensors.items()}, tmp)
        os.replace(tmp, out_path)

    def shard_w13_for_tp(self, t3d: torch.Tensor, tp_rank: int) -> torch.Tensor:
        """
        t3d: (E, per_total, H), per_total = per_tp * (TP*DP)
        Return shard for EP-rank=tp_rank (reusing the function name).
        """
        collapse_mode = self.w13_cols_mode
        E, per_total, H = t3d.shape
        T = self.infer_tp
        if per_total % T != 0:
            raise ValueError(f"per_total={per_total} must be divisible by TP_eff={T}")
        per_tp = per_total // T
        if collapse_mode == "grouped":
            s = tp_rank * per_tp
            e = (tp_rank + 1) * per_tp
            return t3d[:, s:e, :]
        elif collapse_mode == "interleaved":
            return t3d[:, tp_rank::T, :]  # stride by T along per_total
        else:
            raise ValueError(f"Unknown collapse mode: {collapse_mode}")

    def shard_w13_for_tp_with_axi(self, t3d: torch.Tensor, tp_rank: int, split_axis: int = 1) -> torch.Tensor:
        """
        t3d: (E, per_total, H), per_total = per_tp * (TP*DP)
        Return shard for EP-rank=tp_rank (reusing the function name).
        """
        collapse_mode = self.w13_cols_mode
        R = self._Teff
        dim = t3d.shape[split_axis]

        if dim % R != 0:
            raise ValueError(f"w13: dim {split_axis}={dim} not divisible by TP_eff={R}")

        splitted_dim = dim // R
        if collapse_mode == "grouped":
            s = tp_rank * splitted_dim
            e = (tp_rank + 1) * splitted_dim
            slices = [slice(None)] * t3d.ndim
            slices[split_axis] = slice(s, e)
            out = t3d[tuple(slices)]
            return out
        elif collapse_mode == "interleaved":
            slices = [slice(None)] * t3d.ndim
            slices[split_axis] = slice(tp_rank, None, R)
            out = t3d[tuple(slices)]
            return out
        else:
            raise ValueError(f"Unknown collapse mode: {collapse_mode}")

    def shard_w2_for_tp(self, t3d: torch.Tensor, tp_rank: int, split_axis: int = 2) -> torch.Tensor:
        """
        t3d: typically (E, H, per_total) after unflatten → split along `split_axis`
        Applies grouped/interleaved semantics governed by `self.w2_rows_mode`.
        """
        collapse_mode = self.w2_rows_mode
        R = self._Teff
        dim = t3d.shape[split_axis]
        splitted_dim = dim // R
        if dim % R != 0:
            raise ValueError(f"w2: dim {split_axis}={dim} not divisible by EP={R}")
        slices = [slice(None)] * t3d.ndim
        if collapse_mode == "grouped":
            s = tp_rank * splitted_dim
            e = s + splitted_dim
            slices[split_axis] = slice(s, e)
        elif collapse_mode == "interleaved":
            slices[split_axis] = slice(tp_rank, None, R)
        else:
            raise ValueError(f"Unknown w2 mode: {collapse_mode}")
        return t3d[tuple(slices)]

    def _w2_interleave_rows(self, cat: torch.Tensor, ln: int) -> torch.Tensor:
        """
        cat: [ln*per_tp*T, H] after concatenating TP shards along dim=0 for this EP window
        ln : number of experts covered by this window (the 'length' from metadata)

        Return the same shape but with rows interleaved across TP inside each expert:
        for each per_idx: [tp0, tp1, ..., tp(T-1)]
        """
        if self.w2_rows_mode == "grouped" or self.infer_tp == 1:
            return cat
        T = self._Teff
        if R % (ln * T) != 0:
            raise ValueError(f"w2 rows {R} not divisible by ln*TP_eff={ln * T}")
        per_tp = R // (ln * T)
        # [T, ln, per_tp, H] -> [ln, per_tp, T, H] -> flatten per_tp,T with T as the fast index
        x = cat.view(T, ln, per_tp, H).permute(1, 2, 0, 3).contiguous()  # [ln, per_tp, T, H]
        return x.view(ln * per_tp * T, H).contiguous()

    def final_rank_split(self, tensors: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Produce EP = TP * DP groups:
        - experts (w13/w2): split across EP ranks
        - non-experts: split by TP (when applicable) and replicate across DP
        """

        R = self._Teff  # total EP ranks
        T = self.infer_tp
        split_tensors = [{} for _ in range(T)]
        for name, tensor in tensors.items():
            axis = self.get_tp_split_axis(name)
            if self.is_w13(name) and self.has_moe:
                # w13 is (E, per_total, H). Split with stride when interleaved.
                for r in range(T):
                    split_tensors[r][name] = self.shard_w13_for_tp(tensor, r)
                continue
            if self.is_w2(name) and self.has_moe:
                if self.w2_rows_mode == "interleaved":
                    # stride by EP instead of TP
                    for r in range(T):
                        split_tensors[r][name] = tensor[:, r::T, :]
                    continue
                else:
                    per_total = tensor.shape[1]
                    if per_total % T != 0:
                        raise ValueError(f"{name}: per_total={per_total} not divisible by EP={R} in grouped mode")
                    per_rank = per_total // T
                    for r in range(T):
                        s = r * per_rank
                        e = s + per_rank
                        split_tensors[r][name] = tensor[:, s:e, :]
                    continue

            if axis is None or self.infer_tp <= 1:
                piece = tensor
                if self.is_fused_qkv_weight(name) or self.is_fused_qkv_bias(name):
                    piece = self.reshape_qkv_megatron_local(name, piece)
                # replicate to all EP ranks
                for r in range(T):
                    split_tensors[r][name] = piece
            else:
                if tensor.shape[axis] % self.infer_tp != 0:
                    raise ValueError(f"{name}: dim {axis}={tensor.shape[axis]} not divisible by TP={self.infer_tp}")
                shards = list(torch.chunk(tensor, self.infer_tp, dim=axis))
                # map EP rank -> corresponding TP shard (replicate across DP)
                for r in range(T):
                    tp = r % T
                    piece = shards[tp].contiguous()
                    if self.is_fused_qkv_weight(name) or self.is_fused_qkv_bias(name):
                        piece = self.reshape_qkv_megatron_local(name, piece)
                    split_tensors[r][name] = piece

        return split_tensors

    def final_rank_split_new(self, tensors: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Produce EP = TP * DP groups:
        - experts (w13/w2): split across EP ranks
        - non-experts: split by TP (when applicable) and replicate across DP
        """

        R = self._Teff  # total EP ranks
        T = self.infer_tp
        split_tensors = [{} for _ in range(R)]
        for name, tensor in tensors.items():
            axis = self.get_tp_split_axis(name)
            if self.is_w13(name):
                # w13 is (E, per_total, H). Split with stride when interleaved.
                for r in range(R):
                    split_tensors[r][name] = self.shard_w13_for_tp_with_axi(tensor, r,
                                                                            split_axis=axis if axis is not None else 1)
                continue
            if self.is_w2(name):
                # Axis-generalized split for w2 (default axis=2 keeps old behavior)
                split_ax = axis if axis is not None else 2
                for r in range(R):
                    split_tensors[r][name] = self.shard_w2_for_tp(tensor, r, split_axis=split_ax)
                continue

            if axis is None or self.infer_tp <= 1:
                piece = tensor
                if self.is_fused_qkv_weight(name) or self.is_fused_qkv_bias(name):
                    piece = self.reshape_qkv_megatron_local(name, piece)
                # replicate to all EP ranks
                for r in range(T):
                    split_tensors[r][name] = piece
            else:
                if tensor.shape[axis] % self.infer_tp != 0:
                    raise ValueError(f"{name}: dim {axis}={tensor.shape[axis]} not divisible by TP={self.infer_tp}")
                shards = list(torch.chunk(tensor, self.infer_tp, dim=axis))
                # map EP rank -> corresponding TP shard (replicate across DP)
                for r in range(T):
                    tp = r % T
                    piece = shards[tp].contiguous()
                    if self.is_fused_qkv_weight(name) or self.is_fused_qkv_bias(name):
                        piece = self.reshape_qkv_megatron_local(name, piece)
                    split_tensors[r][name] = piece

        return split_tensors

    def assemble_split_write_core(
            self,
            pp2ep_paths: Dict[int, Dict[int, List[Tuple[int, str]]]],
            rc: _ReaderCache, final_dir: str,
            names: Optional[List[str]] = None,
            pid: Optional[str] = None
    ) -> List[str]:
        init_time = time.time()
        names_filter = set(names) if names else None

        tensors = self.assemble_dir(pp2ep_paths, rc, names_filter=names_filter)
        for name, t in tensors.items():
            print(f"ASSEMBLE {name}: {t.shape}")
        assemble_time = time.time()
        if self.use_simple_ep_mode:
            split_tensors = self.final_rank_split_new(tensors)
        else:
            split_tensors = self.final_rank_split(tensors)
        for split_tensor in split_tensors:
            for name, t in split_tensor.items():
                print(f"SPLIT {name}: {t.shape}")
        split_time = time.time()
        logger.info(f"Finished assembling {len(tensors)} tensors in {assemble_time - init_time:.2f}s, "
                    f"splitting EP (TPxDP) in {split_time - assemble_time:.2f}s.")

        written: List[str] = []
        suffix = f"_{pid}" if pid is not None else ""
        for ep in range(len(split_tensors)):  # EP = TP * DP
            out_path = os.path.join(final_dir, f"rank_{ep}{suffix}.safetensors")
            tmp = out_path + ".tmp"
            save_file({k: v.contiguous() for k, v in split_tensors[ep].items()}, tmp)
            os.replace(tmp, out_path)
            written.append(out_path)
        return written


class Qwen3MoEParamsAssembler(ParamsAssembler):
    def __init__(
            self,
            hf_config,
            infer_tp: int = 1,
            infer_dp: int = 1,
            target_dtype: Optional[torch.dtype] = None,
            w13_gate_up_order: str = "gate_up",
            head_dim_scale: int = 1,
            use_simple_ep_mode: bool = False,
    ):
        # w13_gate_up_order ∈ {"gate_up", "up_gate"}; set "up_gate" if your model packs as [up, gate]
        super().__init__(infer_tp=infer_tp, infer_dp=infer_dp, target_dtype=target_dtype, head_dim_scale=head_dim_scale,
                         use_simple_ep_mode=use_simple_ep_mode)
        self.w13_gate_up_order = w13_gate_up_order.lower()
        self._init_configs(hf_config)
        self.has_moe = bool(self.num_experts and self.num_experts > 0)
        self.w2_rows_mode = "grouped"
        self.w13_cols_mode = "interleaved"

    def get_weight_3D_shape(self, name: str, old_shape: List[int]) -> Optional[Tuple]:
        layer_name = self.remove_idx_pattern.sub(".", name)
        if "w13_weight" in layer_name:
            # old: (H, E_total * per)
            old_shape_0 = old_shape[0]
            if old_shape_0 != self.hidden_size:
                raise ValueError(f"w13 expected shape_0={self.hidden_size}, got {old_shape_0}")
            if old_shape[1] % self.num_experts != 0:
                raise ValueError(f"w13 second dim {old_shape[1]} not divisible by E={self.num_experts}")
            per = old_shape[1] // self.num_experts
            return self.hidden_size, self.num_experts, per  # will permute to (E, per, H)
        if "w2_weight" in layer_name:
            # old: (E_total * per, H)
            old_shape1 = old_shape[1]
            if old_shape1 != self.hidden_size:
                raise ValueError(f"w2 expected shape1={self.hidden_size}, got {old_shape1}")
            if old_shape[0] % self.num_experts != 0:
                raise ValueError(f"w2 first dim {old_shape[0]} not divisible by E={self.num_experts}")
            per = old_shape[0] // self.num_experts
            return self.num_experts, per, self.hidden_size  # will permute to (E, H, per)
        return None

    def get_weight_3D_permute(self, name: str) -> Optional[Tuple]:
        layer_name = self.remove_idx_pattern.sub(".", name)
        if "w13_weight" in layer_name:
            return 1, 2, 0
        if "w2_weight" in layer_name:
            return 0, 2, 1
        return None

    def get_train_tp_concat_axis_2d(self, name: str) -> int | None:
        f = self.remove_idx_pattern.sub(".", name)
        # Experts (2-D view):
        #   w13: (H_shard?, E_local*per)  -> concat along dim=0 (row-like)
        #   w2 : (E_local*per, H_shard?)  -> concat along dim=1 (col-like)
        if self.has_moe:
            if "experts.w13_weight" in f: return 1
            if "experts.w2_weight" in f: return 1
        # Common TP rules for non-experts in Megatron:
        if "embed_tokens.weight" in f: return 0
        if "lm_head.weight" in f: return 0
        if "qkv_proj.weight" in f: return 0  # ColumnParallel
        if ".o_proj.weight" in f: return 1  # RowParallel

        # MLP (dense): ColumnParallel up/gate, RowParallel down
        if any(k in f for k in (".up_proj.weight", ".gate_proj.weight", ".w1.weight", ".w3.weight")): return 0
        if any(k in f for k in (".down_proj.weight", ".w2.weight")): return 1

        # bias rules if present
        if "qkv_proj.bias" in f: return 0
        if ".o_proj.bias" in f: return 0
        # fallback: no TP split
        return None

    def get_tp_split_axis(self, name: str) -> Optional[int]:
        f = self.remove_idx_pattern.sub(".", name)
        if self.has_moe and "w13_weight" in f:
            return 0  # 2  # split on output channels
        if self.has_moe and "w2_weight" in f:
            return 0  # 1  # split on input channels
        if "lm_head" in f:
            return 0
        if "embed_tokens" in f:
            return 0
        if "o_proj" in f:
            return 1
        if "qkv_proj" in f:
            return 0

        # MLP (dense): ColumnParallel up/gate, RowParallel down
        if any(k in f for k in (".up_proj", ".gate_proj", ".w1", ".w3")): return 0
        if any(k in f for k in (".down_proj", ".w2")): return 1
        return None

    def _init_configs(self, hf_config):
        tc = getattr(hf_config, "text_config", None)

        def _get(*keys):
            # try hf_config then text_config
            for k in keys:
                v = getattr(hf_config, k, None)
                if v is not None:
                    return v
                if tc is not None:
                    v = getattr(tc, k, None)
                    if v is not None:
                        return v
            return None

        hidden_size = _get("hidden_size", "n_embd")
        target_vocab_size = _get("vocab_size")

        num_heads = _get("num_attention_heads", "n_head", "num_heads")
        num_kv_heads = _get("num_key_value_heads", "n_kv_heads", "num_kv_head", "multi_query_group_num")
        head_dim = _get("head_dim")
        if head_dim is None and hidden_size and num_heads:
            head_dim = hidden_size // num_heads

        if any(x is None for x in (num_heads, num_kv_heads, head_dim)):
            raise ValueError(
                f"Missing config fields: num_heads={num_heads}, "
                f"num_kv_heads={num_kv_heads}, head_dim={head_dim}. "
                "Check your model config or add manual overrides."
            )
        num_experts = getattr(hf_config, "num_experts", None)
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is None and hasattr(hf_config, "text_config"):
            num_layers = getattr(hf_config.text_config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("hf_config must expose num_hidden_layers")

        self.org_vocab_size = target_vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

    def is_fused_qkv_weight(self, name: str) -> bool:
        return ".self_attn.qkv_proj.weight" in _norm(name)

    def is_fused_qkv_bias(self, name: str) -> bool:
        return ".self_attn.qkv_proj.bias" in _norm(name)

    def is_w13(self, name: str) -> bool:
        return "w13_weight" in _norm(name)

    def is_w2(self, name: str) -> bool:
        return "w2_weight" in _norm(name)


@ray.remote
def assemble_subset_worker(
        *,  # kwargs only
        final_dir: str,
        reduce_idx: int,
        names: List[str],
        pp2ep_paths: Dict[int, Dict[int, List[Tuple[int, str]]]],
        assembler: ParamsAssembler,
        max_open_files: int = 64,
        num_procs: int = 1,
        child_idx_stride: int = 1_000_000,
) -> List[str] | None:
    """
    Run inside one Ray task. Forks up to num_procs children.
    Each child processes a subset of `names`, writes rank_{tp}_{pid}.safetensors.
    """
    # Fast single-process path
    if num_procs <= 1 or len(names) <= 1:
        rc = _ReaderCache(max_open=max_open_files, copy_on_read=False)
        try:
            pid = f"{reduce_idx}_0"
            return assembler.assemble_split_write_core(pp2ep_paths, rc, final_dir, names=names, pid=pid)
        except Exception as e:
            logger.error(f"split error: {e}")
            traceback.print_exc()
        finally:
            rc.close_all()

    # Distribute "big" parameters to different procs first, then round-robin the rest
    names = list(names)
    big = [n for n in names if n.endswith("embed_tokens.weight") or n == "lm_head.weight"]
    rest = [n for n in names if n not in big]

    chunks = [[] for _ in range(min(num_procs, len(names)))]
    for i, n in enumerate(big):  chunks[i % len(chunks)].append(n)
    for i, n in enumerate(rest): chunks[i % len(chunks)].append(n)
    chunks = [c for c in chunks if c]

    # Fork children
    ctx = mp.get_context("fork")
    ret_q: mp.Queue = ctx.Queue()
    procs: List[mp.Process] = []

    def _child(proc_idx: int, chunk_list: List[str], ret_queue: mp.Queue):
        # Each child must have its own ReaderCache and thread caps
        try:
            try:
                os.environ.setdefault("OMP_NUM_THREADS", "2")
                os.environ.setdefault("MKL_NUM_THREADS", "1")
                os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                torch.set_num_threads(16)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            rc_ret = _ReaderCache(max_open=max_open_files, copy_on_read=False)
            try:
                tmp_pid = f"{reduce_idx * child_idx_stride + proc_idx}"
                out = assembler.assemble_split_write_core(pp2ep_paths, rc_ret, final_dir, names=chunk_list, pid=tmp_pid)
                ret_queue.put((proc_idx, out, None))
            finally:
                rc_ret.close_all()
        except Exception as ee:
            logger.error(f"child process error: {ee}")
            traceback.print_exc()
            ret_queue.put((proc_idx, None, repr(ee)))

    for i, chunk in enumerate(chunks):
        p = ctx.Process(target=_child, args=(i, chunk, ret_q))
        p.daemon = False
        p.start()
        procs.append(p)

    all_paths: List[str] = []
    for _ in procs:
        idx, paths, err = ret_q.get()
        if err is not None:
            for p in procs: p.join()
            raise RuntimeError(f"assemble_subset_worker child {idx} failed: {err}")
        all_paths.extend(paths)
    for p in procs: p.join()
    return all_paths


def run_distributed_qwen3_assemble(
        train_save_path: str,
        hf_config: Any,
        infer_tp: int,
        weights_version: int,
        inference_save_path: str,
        infer_dp: int = 1,
        target_dtype: Optional[torch.dtype] = torch.bfloat16,
        pattern: str = "pp*_tp*_ep*.safetensors",
        max_open_files: int = 64,
        num_workers: int = 3,
        num_procs: int = 24,
        head_dim_scale: int = 1,
        use_simple_ep_mode: bool = False,
):
    start_time = time.time()
    final_dir = os.path.join(inference_save_path, f"weights_{weights_version}")
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir, ignore_errors=True)
    os.makedirs(final_dir, exist_ok=True)

    logger.info(f"final_dir: {final_dir}")
    assembler = Qwen3MoEParamsAssembler(hf_config, infer_tp, infer_dp, target_dtype,
                                        w13_gate_up_order="gate_up", head_dim_scale=head_dim_scale,
                                        use_simple_ep_mode=use_simple_ep_mode)
    all_paths = sorted(glob.glob(os.path.join(train_save_path, pattern)))
    if not all_paths:
        raise FileNotFoundError(f"no files match {pattern} under {train_save_path}")

    logger.info(f"all_paths: {all_paths}")
    pp2ep_paths = assembler.group_paths_by_pp_ep_tp(all_paths)

    # Discover all final keys once
    rc_list = _ReaderCache(max_open=max_open_files, copy_on_read=False)
    try:
        all_keys = list_all_keys(pp2ep_paths, rc_list)
    finally:
        rc_list.close_all()

    # Fallback single-process if requested
    if num_workers <= 1 and num_procs <= 1:
        rc = _ReaderCache(max_open=max_open_files, copy_on_read=False)
        try:
            assembler.assemble_split_write_core(pp2ep_paths, rc, final_dir, names=None, pid=None)
        finally:
            rc.close_all()
        logger.info(f"single process wrote tensors to {final_dir} succeed, took {time.time() - start_time:.2f}s.")
        return

    # Ensure Ray is up (ok if already initialized from your driver)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Launch Ray workers over the key list
    lg = launch_chunked_with_pg(
        items=all_keys,
        workers=max(1, num_workers),
        ideal_cpus_per_worker=64,  # tune if your CPUs differ
        make_kwargs=lambda idx, chunk: dict(
            final_dir=final_dir,
            reduce_idx=idx,
            names=chunk,
            pp2ep_paths=pp2ep_paths,
            assembler=assembler,  # pickled to the worker
            max_open_files=max_open_files,
            num_procs=max(1, num_procs),
        ),
        flatten_list_results=True,
        use_pg_if_one=False,
    )
    lg.get()  # paths, if you need them
    logger.info(f"wrote tensors to {final_dir}, took {time.time() - start_time:.2f}s.")
    return weights_version

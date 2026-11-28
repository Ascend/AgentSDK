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

from collections.abc import Mapping, Sequence
from typing import Any, Iterator

from omegaconf import OmegaConf


VERL_DEFAULT = "verl_default"
AURA_DEFAULT = "aura_default"
USER_EXPLICIT = "user_explicit"


def build_source_map(
    *,
    verl_default: Any = None,
    aura_default: Any = None,
    user_explicit: Any = None,
    namespace: str = "train_config",
) -> dict[str, str]:
    """Mark each explicitly present leaf using low-to-high merge precedence."""
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")

    sources: dict[str, str] = {}
    layers = (
        (VERL_DEFAULT, verl_default),
        (AURA_DEFAULT, aura_default),
        (USER_EXPLICIT, user_explicit),
    )
    for source, config in layers:
        if config is None:
            continue
        container = _to_plain_container(config)
        if not isinstance(container, Mapping):
            raise TypeError(f"{source} configuration layer must be a mapping")
        for path in _iter_leaf_paths(container):
            full_path = f"{namespace}.{path}" if path else namespace
            sources[full_path] = source
    return sources


def merge_source_maps(*source_maps: Mapping[str, str]) -> dict[str, str]:
    """Merge independently built namespace maps and reject conflicting labels."""
    merged: dict[str, str] = {}
    for source_map in source_maps:
        for path, source in source_map.items():
            if path in merged and merged[path] != source:
                raise ValueError(f"Conflicting source labels for {path}: {merged[path]} and {source}")
            merged[path] = source
    return merged


def _to_plain_container(config: Any) -> Any:
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=False)
    return config


def _iter_leaf_paths(value: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(value, Mapping):
        if not value:
            yield prefix
            return
        for key, child in value.items():
            key_path = str(key) if not prefix else f"{prefix}.{key}"
            yield from _iter_leaf_paths(child, key_path)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        yield prefix
        if not value:
            return
        for index, child in enumerate(value):
            index_path = str(index) if not prefix else f"{prefix}.{index}"
            yield from _iter_leaf_paths(child, index_path)
        return

    yield prefix

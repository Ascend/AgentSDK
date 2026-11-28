# SPDX-License-Identifier: MulanPSL-2.0
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
"""Resolve the committed model checkpoint shared by trainer and rollout."""

import os


def checkpoint_version(path):
    name = os.path.basename(os.path.normpath(path))
    prefix = "global_step_"
    if not name.startswith(prefix) or not name[len(prefix):].isdigit():
        raise ValueError("resume ckpt must specify the global_steps")
    return int(name[len(prefix):])


def resolve_checkpoint(trainer_config):
    mode = trainer_config.resume_mode
    if mode == "disable":
        return None
    if mode not in ("auto", "resume_path"):
        raise ValueError(f"Unsupported resume_mode: {mode}")
    if trainer_config.default_hdfs_dir is not None:
        raise NotImplementedError("load from hdfs is not implemented yet")
    if mode == "resume_path":
        path = trainer_config.resume_from_path
        if not isinstance(path, str):
            raise ValueError("resume ckpt must be str type")
    else:
        root = os.path.abspath(trainer_config.default_local_dir)
        pointer = os.path.join(root, "latest_checkpointed_iteration.txt")
        if not os.path.isfile(pointer):
            return None
        with open(pointer, encoding="utf-8") as stream:
            version = stream.read().strip()
        if not version.isdigit():
            raise ValueError("Invalid latest_checkpointed_iteration.txt")
        path = os.path.join(root, f"global_step_{int(version)}")
    path = os.path.abspath(os.path.normpath(path))
    checkpoint_version(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path

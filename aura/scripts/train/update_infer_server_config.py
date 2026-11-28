#!/usr/bin/env python3
# coding=utf-8
"""Update external inference endpoints in a training YAML atomically."""

import argparse
import os
import stat
import tempfile
from pathlib import Path

import yaml


def _parse_server_list(value: str) -> list[str]:
    if not value.strip():
        return []
    servers = yaml.safe_load(f"[{value}]")
    if not isinstance(servers, list) or not all(isinstance(server, str) for server in servers):
        raise ValueError(f"Invalid inference server list: {value!r}")
    return servers


def _parse_bool(value: str) -> bool:
    parsed = yaml.safe_load(value)
    if not isinstance(parsed, bool):
        raise ValueError(f"Expected a boolean value, got: {value!r}")
    return parsed


def update_config(args: argparse.Namespace) -> None:
    config_path = Path(args.config_file)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    infer_instances = config.get("infer_instances")
    if not isinstance(infer_instances, list) or not infer_instances:
        raise ValueError("infer_instances must be a non-empty list")

    prefill_servers = _parse_server_list(args.prefill_server_list)
    decode_servers = _parse_server_list(args.decode_server_list)
    enable_expert_parallel = _parse_bool(args.enable_expert_parallel)

    updated = 0
    for instance in infer_instances:
        engine_kwargs = instance.get("executor_kwargs", {}).get("engine_kwargs")
        if not isinstance(engine_kwargs, dict):
            continue
        engine_kwargs.update(
            chat_server=f"http://{args.rollout_host}:8080",
            prefill_server_list=prefill_servers,
            decode_server_list=decode_servers,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
        )
        updated += 1

    if updated == 0:
        raise ValueError("No infer instance contains executor_kwargs.engine_kwargs")

    original_mode = stat.S_IMODE(config_path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.", suffix=".tmp", dir=config_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        yaml.safe_load(temporary_path.read_text(encoding="utf-8"))
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, config_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"[SUMMARY] inference_config_update=passed instances={updated} file={config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--rollout-host", required=True)
    parser.add_argument("--prefill-server-list", required=True)
    parser.add_argument("--decode-server-list", required=True)
    parser.add_argument("--tensor-parallel-size", required=True, type=int)
    parser.add_argument("--data-parallel-size", required=True, type=int)
    parser.add_argument("--enable-expert-parallel", required=True)
    update_config(parser.parse_args())


if __name__ == "__main__":
    main()

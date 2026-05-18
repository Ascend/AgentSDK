import os
import torch
import argparse


def rank_dirs(tp: int, pp: int, ep: int):
    """Generate all rank directory names based on tensor, pipeline, and expert parallelism."""
    dirs = []
    for tp_rank in range(tp):
        for pp_rank in range(pp):
            for ep_rank in range(ep):
                dirs.append(f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}")
    return dirs


def compare_checkpoints(source_checkpoint: str, target_checkpoint: str, tp: int, pp: int, ep: int):
    """
    Compare parameters between two Megatron-format checkpoints across all ranks.

    Args:
        source_checkpoint (str): Root directory of the source Megatron checkpoint.
        target_checkpoint (str): Root directory of the target Megatron checkpoint.
        tp (int): Tensor parallel size.
        pp (int): Pipeline parallel size.
        ep (int): Expert parallel size.
    """
    rank_list = rank_dirs(tp, pp, ep)

    total_keys = 0
    same_keys = 0

    for rank in rank_list:
        file1 = os.path.join(source_checkpoint, rank, "model_optim_rng.pt")
        file2 = os.path.join(target_checkpoint, rank, "model_optim_rng.pt")

        if not os.path.exists(file1):
            raise FileNotFoundError(f"Missing file in source checkpoint: {file1}")
        if not os.path.exists(file2):
            raise FileNotFoundError(f"Missing file in target checkpoint: {file2}")

        print(f"--- Loading {rank} ---", flush=True)
        a = torch.load(file1, weights_only=False)
        b = torch.load(file2, weights_only=False)
        print(f"--- Finished loading {rank} ---", flush=True)

        print(f"\n=== Comparing {rank} ===", flush=True)
        for key in a["model"].keys():
            if a["model"][key] is not None and b["model"][key] is not None:
                tensor_a = a["model"][key].to("cpu")
                tensor_b = b["model"][key].to("cpu")
                is_equal = torch.equal(tensor_a, tensor_b)
                total_keys += 1
                if is_equal:
                    same_keys += 1
                    print(f"{key}: identical", flush=True)
                else:
                    diff = (tensor_a - tensor_b).abs().max().item()
                    print(f"{key}: mismatch, max difference {diff}", flush=True)
            else:
                print(f"{key}: one of the tensors is None", flush=True)

    print(f"\nSummary: {same_keys}/{total_keys} parameters are identical", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare parameter differences between two Megatron-format checkpoints"
    )
    parser.add_argument(
        "--source-checkpoint", type=str, required=True, help="Root directory of the source Megatron checkpoint"
    )
    parser.add_argument(
        "--target-checkpoint", type=str, required=True, help="Root directory of the target Megatron checkpoint"
    )
    parser.add_argument("--tensor-parallel-size", type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel-size", type=int, required=True, help="Pipeline parallel size")
    parser.add_argument("--expert-parallel-size", type=int, required=True, help="Expert parallel size")

    args = parser.parse_args()

    compare_checkpoints(
        args.source_checkpoint,
        args.target_checkpoint,
        args.tensor_parallel_size,
        args.pipeline_parallel_size,
        args.expert_parallel_size,
    )

import argparse
import asyncio
import os

import pandas as pd

from rllm.agents import Trajectory
from rllm.dataset import DatasetRegistry


def load_problems(num_samples):
    dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    if dataset is None:
        raise RuntimeError(
            "Dataset 'deepscaler_math' not found. Run prepare_math_data.py first to register the dataset."
        )

    data = dataset.get_data()
    if num_samples < len(data):
        df = pd.DataFrame(data)
        df = df.sample(n=num_samples, random_state=42)
        data = df.to_dict("records")

    return [{"question": row["question"], "ground_truth": row["ground_truth"], "uid": i} for i, row in enumerate(data)]


async def generate_trajectories(tasks) -> list[Trajectory]:
    """Generate trajectories using DeepScaleR-1.5B."""
    from transformers import AutoTokenizer

    from rllm.agents import MathAgent
    from aura.runner.agent_engine_wrapper.rllm.agent_execution_engine import AsyncAgentExecutionEngine
    from rllm.environments.base.single_turn_env import SingleTurnEnvironment
    from rllm.rewards import math_reward_fn

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model_name = "/home/work/weights/Qwen_QWQ-32B/"

    engine = AsyncAgentExecutionEngine(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args={"reward_fn": math_reward_fn},
        engine_name="openai",
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        sampling_params={"temperature": 0.6, "top_p": 0.95, "model": model_name},
        rollout_engine_args={"base_url": "http://localhost:8000/v1", "api_key": "None"},
        max_response_length=15000,
        max_prompt_length=2048,
        n_parallel_agents=256,
    )

    return await engine.execute_tasks(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    # Load and generate
    tasks = load_problems(args.num_samples)
    results = asyncio.run(generate_trajectories(tasks))
    print([len(r.steps) for r in results])
    print(results)


if __name__ == "__main__":
    main()

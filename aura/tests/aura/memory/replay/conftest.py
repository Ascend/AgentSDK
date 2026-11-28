from dataclasses import replace

import pytest

from aura.memory.replay.schema import ReplaySample, ReplayTensor


@pytest.fixture
def make_group():
    def factory(store_id="store", group_id="g", *, reward=0.9, version=3, eligible=True):
        return [ReplaySample(
            source_store_id=store_id, trajectory_id=f"{group_id}/trajectory/{index}",
            prompt_id=f"prompt-{group_id}", group_id=group_id, sample_index=index, group_size=2,
            source_experiment_id="experiment-a", source_run_id="run-a", policy_lineage_id="lineage-a",
            weight_version=version, reward=reward,
            payload={"prompts": ReplayTensor.from_values([10, 11]),
                     "responses": ReplayTensor.from_values([20, 21, 22]),
                     "response_mask": ReplayTensor.from_values([True, False, True], "bool"),
                     "token_level_rewards": ReplayTensor.from_values([0.0, 0.0, reward], "float32")},
            old_log_probs=ReplayTensor.from_values([-0.25, 0.0, -0.5], "float32") if eligible else None,
            model_signature="model-a", tokenizer_signature="tokenizer-template-a",
            task_signature="gsm8k/train", reward_signature="math-v1", source_batch_id=f"batch-{group_id}",
            train_eligible=eligible, invalid_reason=None if eligible else "incomplete-generation",
        ) for index in range(2)]
    return factory

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

from datasets import load_dataset

from rllm.dataset import DatasetRegistry


def prepare_math_data():
    test_dataset = load_dataset("/home/work/dataset/tmp", split="train")

    def preprocess_fn(example, idx):
        return {
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "data_source": "ioh",
        }

    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    test_dataset = DatasetRegistry.register_dataset("dtn_data", test_dataset, "test")
    return test_dataset


if __name__ == "__main__":
    test_dataset = prepare_math_data()
    print(test_dataset)

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

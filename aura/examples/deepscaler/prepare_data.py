from datasets import load_dataset

from rllm.dataset import DatasetRegistry


def prepare_math_data():
    train_dataset = load_dataset("/home/work/examples/deepscaler/DeepScaleR-Preview-Dataset/", split="train")

    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": "math",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)

    train_dataset = DatasetRegistry.register_dataset("deepscaler_math", train_dataset, "train")
    return train_dataset


if __name__ == "__main__":
    train_dataset = prepare_math_data()
    print(train_dataset)

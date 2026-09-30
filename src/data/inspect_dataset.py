from datasets import load_dataset
from pprint import pprint


def main():
    print("Loading RAGTruth dataset...\n")

    dataset = load_dataset("wandb/RAGTruth-processed")

    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(dataset)

    print("\n" + "=" * 70)
    print("AVAILABLE SPLITS")
    print("=" * 70)

    print(dataset.keys())

    # Use train split for inspection
    train_data = dataset["train"]

    print("\n" + "=" * 70)
    print("NUMBER OF TRAIN EXAMPLES")
    print("=" * 70)

    print(len(train_data))

    print("\n" + "=" * 70)
    print("COLUMN NAMES")
    print("=" * 70)

    print(train_data.column_names)

    print("\n" + "=" * 70)
    print("FIRST EXAMPLE")
    print("=" * 70)

    first_example = train_data[0]
    pprint(first_example)

    print("\n" + "=" * 70)
    print("FIELD TYPES")
    print("=" * 70)

    for key, value in first_example.items():
        print(f"{key}: {type(value).__name__}")


if __name__ == "__main__":
    main()
import argparse
from email.policy import default


def parse_opt():
    parser = argparse.argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default="./data/",
        help="Path to the train data",
    )
    parser.add_argument(
        "--test_data",
        default="./data/",
        help="Path to the test data",
    )
    parser.add_argument(
        "--max_lenght",
        type=int,
        default=64,
        help="Max lenght for tokenizer to be considered.",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Train batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Test batch size."
    )

    parser.add_argument(
        "--epoch", type=int, default=50, help="Number of epochs for training."
    )
    args = parser.parse_args(args=[])
    return args

import os
import argparse
import logging
from glob import glob
import math
import random

def main():
    """Main function of the script."""

    SEED = 42

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", help="All the datasets to combine")
    parser.add_argument("--split_size", type=int, help="Percentage to use as Testing data")
    parser.add_argument("--training_data_output", type=str, help="path to training output data")
    parser.add_argument("--testing_data_output", type=str, help="path to testing output data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.datasets)
    print("Split size:", args.split_size)
    print("Training folder:", args.training_data_output)
    print("Testing folder:", args.testing_data_output)

    train_test_split_factor = args.split_size / 100
    datasets = args.datasets

    print( datasets)
    
    training_datapaths = []
    testing_datapaths = []

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        print(glob(dataset))
        print(glob(os.path.join(dataset, "*")))
        print(glob(os.path.join(dataset, "*/*.jpg")))
        print(glob(os.path.join(dataset, "*/*.jpg"))[:5])
        food_images = glob(os.path.join(dataset, "*/*.jpg"))  # Assuming the images are in subfolders named after classes
        print(f"Found {len(food_images)} images for {dataset}")

        random.seed(SEED)
        random.shuffle(food_images)

        amount_of_test_images = math.ceil(len(food_images) * train_test_split_factor)

        food_test_images = food_images[:amount_of_test_images]
        food_training_images = food_images[amount_of_test_images:]

        testing_datapaths.extend(food_test_images)
        training_datapaths.extend(food_training_images)

        print(testing_datapaths[:5])

        for img in food_test_images:
            with open(img, "rb") as f:
                with open(os.path.join(args.testing_data_output, os.path.basename(img)), "wb") as f2:
                    f2.write(f.read())

        for img in food_training_images:
            with open(img, "rb") as f:
                with open(os.path.join(args.training_data_output, os.path.basename(img)), "wb") as f2:
                    f2.write(f.read())

if __name__ == "__main__":
    main()

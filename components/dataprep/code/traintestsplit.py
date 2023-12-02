import os
import argparse
from glob import glob
import math
import random
import shutil

def main():
    SEED = 42

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

    training_datapaths = []
    testing_datapaths = []

    for dataset in datasets:
        class_folders = glob(os.path.join(dataset, "*"))

        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
            class_images = glob(os.path.join(class_folder, "*.jpg"))

            random.seed(SEED)
            random.shuffle(class_images)

            amount_of_test_images = math.ceil(len(class_images) * train_test_split_factor)

            class_test_images = class_images[:amount_of_test_images]
            class_training_images = class_images[amount_of_test_images:]

            testing_datapaths.extend(class_test_images)
            training_datapaths.extend(class_training_images)

            for img in class_test_images:
                shutil.copy(img, os.path.join(args.testing_data_output, class_name, os.path.basename(img)))

            for img in class_training_images:
                shutil.copy(img, os.path.join(args.training_data_output, class_name, os.path.basename(img)))

if __name__ == "__main__":
    main()

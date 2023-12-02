import os
import argparse
from glob import glob
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--output_data", type=str, help="path to output data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    print("output folder:", args.output_data)

    output_dir = args.output_data
    size = (100, 100)  # You can adjust the size as needed

    # Iterate through train, test, validation folders
    for split_folder in ["train", "test", "validation"]:
        input_split_folder = os.path.join(args.data, split_folder)
        output_split_folder = os.path.join(output_dir, split_folder)

        # Iterate through class folders
        for class_folder in glob(os.path.join(input_split_folder, "*")):
            class_name = os.path.basename(class_folder)

            # Create corresponding output class folder
            output_class_folder = os.path.join(output_split_folder, class_name)
            os.makedirs(output_class_folder, exist_ok=True)

            # Iterate through images in class folder
            for file in glob(os.path.join(class_folder, "*.jpg")):
                img = Image.open(file)
                img_resized = img.resize(size)

                # Save the resized image to the output directory
                output_file = os.path.join(output_class_folder, os.path.basename(file))
                img_resized.convert("RGB").save(output_file)

                # Create a flipped version of the image
                flipped_img = np.fliplr(img_resized)
                output_file = os.path.join(output_class_folder, "flipped_" + os.path.basename(file))
                flipped_img.convert("RGB").save(output_file)

            
if __name__ == "__main__":
    main()

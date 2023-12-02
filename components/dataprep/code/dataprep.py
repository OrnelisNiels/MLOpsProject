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

    # Data augmentation settings (random rotations and flips)
    datagen = ImageDataGenerator(
        rotation_range=180,        # Random rotations from 0 to 180 degrees
        horizontal_flip=True,      # Random horizontal flips
        vertical_flip=True,        # Random vertical flips
        channel_shift_range=0,     # Ensure the number of channels remains the same
    )

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
                # img_resized.convert("RGB").save(output_file)

                # # Augment the resized image
                # augmented_images = datagen.flow(img_resized)[0] 

                # # Save the augmented images
                # for i, augmented_image in enumerate(augmented_images):
                #     output_filename = f"augmented_{i}_{os.path.basename(file)}"
                #     output_file = os.path.join(output_class_folder, output_filename)
                #     augmented_image = Image.fromarray(augmented_image.astype('uint8'))
                #     augmented_image.convert("RGB").save(output_file)
               
               
                # Convert the resized image to HSV color space
                img_resized_hsv = np.array(img_resized.convert("HSV"))

                # Save the HSV image
                output_filename = f"hsv_{os.path.basename(file)}"
                output_file = os.path.join(output_class_folder, output_filename)
                Image.fromarray(img_resized_hsv.astype('uint8'), 'HSV').save(output_file)




if __name__ == "__main__":
    main()

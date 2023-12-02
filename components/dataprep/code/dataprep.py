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
                img_resized.convert("RGB").save(output_file)

                # Apply data augmentation and save augmented images
                augmented_images = []
                img_array = np.array(img_resized)
                img_array = img_array.reshape((1,) + img_array.shape)
                for batch in datagen.flow(img_array, batch_size=1):
                    augmented_images.append(batch[0].astype(np.uint8))
                    if len(augmented_images) >= 2:  # Adjust the number of augmentations as needed
                        break

                for i, augmented_img in enumerate(augmented_images):
                    augmented_img_pil = Image.fromarray(augmented_img)
                    augmented_output_file = os.path.join(output_class_folder, f"aug_{i}_{os.path.basename(file)}")
                    augmented_img_pil.convert("RGB").save(augmented_output_file)

if __name__ == "__main__":
    main()

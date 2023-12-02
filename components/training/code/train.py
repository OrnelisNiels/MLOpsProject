import argparse
import os
from glob import glob
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from azureml.core import Run
from utils import *

SEED = 42
INITIAL_LEARNING_RATE = 0.01
BATCH_SIZE = 32
PATIENCE = 11
model_name = 'food-cnn'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_folder', type=str, dest='training_folder', help='training folder mounting point')
    parser.add_argument('--testing_folder', type=str, dest='testing_folder', help='testing folder mounting point')
    parser.add_argument('--output_folder', type=str, dest='output_folder', help='Output folder')
    parser.add_argument('--epochs', type=int, dest='epochs', help='The amount of Epochs to train')
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    training_folder = args.training_folder
    print('Training folder:', training_folder)

    testing_folder = args.testing_folder
    print('Testing folder:', testing_folder)

    output_folder = args.output_folder
    print('Testing folder:', output_folder)

    MAX_EPOCHS = args.epochs

    training_paths = glob(os.path.join(training_folder, "*/*.jpg"), recursive=True)
    testing_paths = glob(os.path.join(testing_folder, "*/*.jpg"), recursive=True)

    print("Training samples:", len(training_paths))
    print("Testing samples:", len(testing_paths))

    random.seed(SEED)
    random.shuffle(training_paths)
    random.seed(SEED)
    random.shuffle(testing_paths)

    print(training_paths[:3])  # Examples
    print(testing_paths[:3])  # Examples

    X_train = getFeatures(training_paths)
    y_train = getTargets(training_paths)

    X_test = getFeatures(testing_paths)
    y_test = getTargets(testing_paths)

    print('Shapes:')
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_test))

    LABELS, y_train, y_test = encodeLabels(y_train, y_test)
    print('One Hot Shapes:')
    print(y_train.shape)
    print(y_test.shape)

    model_path = os.path.join(output_folder, model_name)
    os.makedirs(model_path, exist_ok=True)

    cb_save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                            monitor='val_loss',
                                                            save_best_only=True,
                                                            verbose=1)

    cb_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=PATIENCE,
                                                     verbose=1,
                                                     restore_best_weights=True)

    cb_reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=4, verbose=1)

    opt = SGD(lr=INITIAL_LEARNING_RATE, decay=INITIAL_LEARNING_RATE / MAX_EPOCHS)

    model = buildModel((64, 64, 3), len(LABELS))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    history = model.fit(aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(X_test, y_test),
                        steps_per_epoch=len(X_train) // BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        callbacks=[cb_save_best_model, cb_early_stop, cb_reduce_lr_on_plateau])

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=LABELS))

    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print(cf_matrix)

    np.save(os.path.join(output_folder, 'confusion_matrix.npy'), cf_matrix)

    print("DONE TRAINING")


if __name__ == "__main__":
    main()

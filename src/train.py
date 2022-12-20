from pathlib import Path
import tensorflow as tf
from dvc.api import params_show
import pandas as pd

# Set the paths to the train and validation directories
BASE_DIR = Path(__file__).parent.parent
data_dir = BASE_DIR / "data"

# Extract the parameters
params = params_show()["train"]
IMAGE_WIDTH, IMAGE_HEIGHT = params["image_width"], params["image_height"]

# Create an ImageDataGenerator object for the train set with augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    fill_mode="nearest",
)

train_generator = train_datagen.flow_from_directory(
    data_dir / "prepared" / "train",
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=params["batch_size"],
    class_mode="categorical",
)

# Do the same for test
test_dataget = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_dataget.flow_from_directory(
    data_dir / "prepared" / "test",
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=params["batch_size"],
    class_mode="categorical",
)


def get_model():
    """Define the model to be fit"""
    # Define a CNN model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                activation="relu",
                input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
            ),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(43, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    return model


def main():
    # Get the model
    model = get_model()
    # Create a path to save the model
    model_path = BASE_DIR / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path / "model.keras", monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5),
    ]
    # Fit the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=params["n_epochs"],
        validation_data=test_generator,
        callbacks=callbacks,
    )

    # Save the metrics
    pd.DataFrame(history.history).to_csv("metrics.csv", index=False)

if __name__ == "__main__":
    main()

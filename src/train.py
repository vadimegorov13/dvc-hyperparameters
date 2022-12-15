from pathlib import Path

import tensorflow as tf
from joblib import dump

# Set the paths to the train and validation directories
base_dir = Path(__file__).parent.parent
data_dir = base_dir / "data"

# Create an ImageDataGenerator object for the train set
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    zoom_range=0.2,  # Randomly zoom in and out of images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode="nearest",  # Fill in missing pixels with nearest neighbor
)

# Generate training data from the train directory
train_generator = data_gen.flow_from_directory(
    data_dir / "raw" / "train",  # Target directory
    target_size=(50, 50),  # Resize images to 150x150
    batch_size=64,  # Set batch size
    class_mode="categorical",  # Use categorical labels
)


def get_model():
    """Define the model to be fit"""
    # Define a CNN model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                                   input_shape=(50, 50, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(43, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        # Use categorical cross-entropy loss
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),  # Use Adam optimizer
        metrics=["accuracy"],  # Calculate accuracy

    )

    return model


def main():
    # Get the model
    model = get_model()

    # Fit the model
    history = model.fit(
        train_generator,  # Use the train generator
        steps_per_epoch=100,
        epochs=10,  # Train for 10 epochs
    )

    metrics_dir = base_dir / "metrics"
    models_dir = base_dir / "models"
    metrics_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    dump(history.history, metrics_dir / "history.joblib")
    dump(model, models_dir / "model.joblib")


if __name__ == "__main__":
    main()

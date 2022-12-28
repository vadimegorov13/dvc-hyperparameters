import shutil
from pathlib import Path
import numpy as np
from dvc.api import params_show

np.random.seed(42)

# Set up the directories
base_dir = Path(__file__).parent.parent
raw_train_dir = base_dir / "data" / "raw" / "train"
raw_test_dir = base_dir / "data" / "raw" / "test"

# Load the parameters from params.yaml
params = params_show()['split']

# Copy 'ratio'% of train images to validation directory
for directory in raw_train_dir.iterdir():
    test_mirror_path = str(directory).replace("train", "test")
    test_mirror_path = Path(test_mirror_path)
    test_mirror_path.mkdir(parents=True, exist_ok=True)

    # Collect image paths in each class of train directory
    image_paths = list(directory.glob("*.png"))
    np.random.shuffle(image_paths)

    # Choose 'ratio'% of images (parameter is loaded from `params.yaml`)
    test_images = image_paths[-int(len(image_paths) * params['ratio']):]

    # Copy images to validation directory
    for image_path in test_images:
        shutil.move(image_path, test_mirror_path)

# Reverse the above operation
# for directory in raw_test_dir.iterdir():
#     train_mirror_path = str(directory).replace("test", "train")
#     train_mirror_path = Path(train_mirror_path)

#     # Collect image paths in each class of validation directory
#     image_paths = list(directory.glob("*.png"))

#     # Copy images to train directory
#     for image_path in image_paths:
#         shutil.move(image_path, train_mirror_path)

import os
import string
import numpy as np
from tqdm import tqdm

from modules import Image2Landmarks

# ----- HELPER FUNCTIONS -------
def drop_null(dataset:dict) -> dict:
    """ Removes None data from the dataset """
    filtered_data, filtered_targets, filtered_paths = zip(*[(d,t,p) for d,t,p in zip(dataset['data'], dataset['target'], dataset['path'])
                                                            if d is not None])
    return {'data': np.array(filtered_data), 'target': filtered_targets, 'path': filtered_paths}
# ------------------------------


# ----- GLOBAL VARIABLES -------
IMAGES_PATH = './images'
OUTPUT_PATH = './resources'
classes = [c for c in string.ascii_uppercase if c not in ("J", "Z")]
# ------------------------------


# ------- Initinating landmarker
image2landmarks = Image2Landmarks(
    model_path="./resources/hand_landmarker.task",
    landmark_type="world",
)

# -------- Extracting Landmarks
dataset = {'data': [], 'target': [], 'path': []}
for c in classes:
    print(f"Extracting hand world landmarks for class {c} ...")

    all_hand_landmarks = []
    image_paths = os.listdir(os.path.join(IMAGES_PATH, c))  # list of all image filenames

    for i, img in tqdm(enumerate(image_paths), desc="   Processing"):
        # add targets/label
        dataset['target'].append(c)

        # add img_path
        img_path = os.path.join(os.path.join(IMAGES_PATH, c), img)
        dataset['path'].append(img_path)

        # extract and add landmarks/data
        hand_landmarks = image2landmarks.image_to_hand_landmarks(img_path)
        dataset['data'].append(hand_landmarks)
    print("   Done! \n")

# save extracted landmarks
np.save('./resources/alphabet_landmarks.npy', dataset)
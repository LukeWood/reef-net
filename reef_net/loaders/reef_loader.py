import json
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def get_bboxes(annotations):
    result = []
    annotations = annotations.replace("'", '"')
    annotations = json.loads(annotations)
    for bbox in annotations:
        box = [bbox["x"], bbox["y"], bbox["width"], bbox["height"]]
        box = [float(x) for x in box]
        result.append(box)
    return result


#
# def parse_annotations(annotations, max_boxes):
#     result = []
#     annotations = annotations.replace("'", '"')
#     annotations = json.loads(annotations)
#     for bbox in annotations:
#         box = [bbox["x"], bbox["y"], bbox["width"], bbox["height"], 1]
#         box = [float(x) for x in box]
#         result.append(box)
#
#     result = tf.constant(result, tf.float32)
#     if len(result.shape) == 1:
#         result = tf.expand_dims(result, axis=0)
#
#     return pad_to_shape(result, (max_boxes, 5))


def load_image(self, idx):
    image_path = self.image_paths[idx]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    return tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    # return tf.image.resize(img, [img_height, img_width])


def load_reef_dataset(config, min_boxes_per_image=1):
    base_path = config.data_path
    csv_path = os.path.abspath(os.path.join(base_path, "train.csv"))
    image_base_path = os.path.abspath(os.path.join(base_path, "train_images"))

    df = pd.read_csv(csv_path)

    df["image_path"] = (
        image_base_path
        + "/video_"
        + df["video_id"].astype(str)
        + "/"
        + df["video_frame"].astype(str)
        + ".jpg"
    )

    df["boxes"] = df["annotations"].apply(lambda annotations: get_bboxes(annotations))
    df["num_bboxes"] = df["boxes"].apply(lambda x: len(x))
    max_boxes = df["num_bboxes"].max()

    # get subset of df with images containing at least 1 bounding box
    df = df[df["num_bboxes"] >= min_boxes_per_image]

    image_paths = df["image_path"]
    boxes = df["boxes"]

    def dataset_generator():
        for image_path, annotations in zip(image_paths, boxes):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            annotations = np.array(annotations)
            category = [[0] * annotations.shape[0]]
            category = list(np.concatenate(category).flat)
            yield (img, annotations, category)

    return tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    ), len(df)

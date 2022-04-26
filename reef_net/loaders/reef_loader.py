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


def pad_to_shape(a, shape):
    for _ in range(len(shape) - len(a.shape)):
        a = np.expand_dims(a, axis=0)
    y_, x_ = shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(a, ((0, y_pad), (0, x_pad)), mode="constant", constant_values=-1)


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
    tf.print(img)
    # Convert the compressed string to a 3D uint8 tensor
    return tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    # return tf.image.resize(img, [img_height, img_width])


def load_reef_dataset(config, min_boxes_per_image=0):
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

    if min_boxes_per_image != 0:
        df = df[df["num_bboxes"] > min_boxes_per_image]

    image_paths = df["image_path"]
    boxes = df["boxes"]

    def dataset_generator():
        for image_path, annotations in zip(image_paths, boxes):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            # img = cv2.resize(img, 640, 640)
            annotations = np.array(annotations)
            annotations = pad_to_shape(annotations, (max_boxes, 4))
            yield (img, np.array(annotations), [0])

    return tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        ),
    )

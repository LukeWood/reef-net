import cv2
import matplotlib.pyplot as plt
import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags
from reef_net.preprocess import preprocess_data
import reef_net
import numpy as np

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file("config", "configs/main.py")


def visualize_bounding_boxes(img, annotations, category):
    for annotation in annotations:
        x, y, w, h = annotation.astype(np.float32)
        x1 = (x-w/2).astype(int)
        x2 = (x+w/2).astype(int)
        y1 = (y-h/2).astype(int)
        y2 = (y+h/2).astype(int)
        if category == -1:
            break
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return img


def main(args):
    config = FLAGS.config
    ds = reef_net.loaders.load_reef_dataset(config, min_boxes_per_image=5)
    ds = ds.shuffle(20)
    ds = ds.map(preprocess_data)

    (image, bounding_boxes, category) = next(iter(ds.take(1)))
    image, bounding_boxes, category = image.numpy(), bounding_boxes.numpy(), category.numpy()
    print("Image size", image.shape)
    image = visualize_bounding_boxes(image, bounding_boxes, category)
    plt.imshow(image / 255.0)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    app.run(main)

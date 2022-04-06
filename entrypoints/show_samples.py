import cv2
import matplotlib.pyplot as plt
import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import reef_net

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file("config", "configs/main.py")


def visualize_bounding_boxes(img, annotations):
    for annotation in annotations:
        x, y, w, h, category = annotation.astype(int)
        if h == -1:
            break
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return img


def main(args):
    config = FLAGS.config
    ds = reef_net.loaders.load_reef_dataset(config)
    ds = ds.shuffle(100)

    (image, labels) = next(iter(ds.take(1)))
    image, labels = image.numpy(), labels.numpy()

    image = visualize_bounding_boxes(image, labels)
    plt.imshow(image / 255.0)
    plt.show()


if __name__ == "__main__":
    app.run(main)

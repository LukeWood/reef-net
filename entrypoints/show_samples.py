import matplotlib.pyplot as plt
import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import reef_net

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file("config", "configs/main.py")


def main(args):
    config = FLAGS.config
    ds = reef_net.loaders.load_reef_dataset(config)

    (image, labels) = next(iter(ds.take(1)))
    image, labels = image.numpy(), labels.numpy()
    plt.imshow(image/255.0)
    plt.show()


if __name__ == "__main__":
    app.run(main)

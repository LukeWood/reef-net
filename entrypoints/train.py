import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "Whether to run to wandb")
flags.DEFINE_string("artifact_dir", None, "Directory to store artifacts")


def get_dataset(config):
    if config.dataset == "cifar10":
        return cifar10_ssl_loader.prepare_autoencoder_datasets(config)
    elif config.dataset == "mnist":
        return mnist_loader.prepare_autoencoder_datasets(config)
    raise ValueError(f"Unsupported dataset in `get_dataset`, got={config.dataset}")


def main(args):
    del args
    config = FLAGS.config
    try:
        os.makedirs(FLAGS.artifact_dir)
    except:
        pass

    strategy = tf.distribute.MirroredStrategy()
    logging.info("Number of devices: {}".format(strategy.num_replicas_in_sync))
    logging.info(f"Running with model_dir: {FLAGS.model_dir}")
    logging.info(f"Running with artifact_dir: {FLAGS.artifact_dir}")

    if FLAGS.wandb:
        wandb.init(
            project=config.wandb_project_name, entity="aacl", config=config.to_dict()
        )


if __name__ == "__main__":
    app.run(main)

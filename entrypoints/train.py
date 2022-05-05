import os
from datetime import datetime

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import wandb
from reef_net.loaders import load_reef_dataset
from reef_net.model import DecodePredictions
from reef_net.model import RetinaNet
from reef_net.model import get_backbone
from reef_net.preprocess import preprocess_data
from reef_net.preprocess import resize_and_pad_image
from reef_net.utils import LabelEncoder
from reef_net.utils import visualize_detections

config_flags.DEFINE_config_file("config", "configs/main.py")

flags.DEFINE_bool("wandb", False, "Whether to run to wandb")
flags.DEFINE_bool("debug", False, "Whether or not to use extra debug utilities")
flags.DEFINE_string("artifact_dir", None, "Directory to store artifacts")
flags.DEFINE_bool("model_dir", False, "Where to save the model after training")

FLAGS = flags.FLAGS


def get_dataset(config):
    if config.dataset == "cifar10":
        return cifar10_ssl_loader.prepare_autoencoder_datasets(config)
    elif config.dataset == "mnist":
        return mnist_loader.prepare_autoencoder_datasets(config)
    raise ValueError(f"Unsupported dataset in `get_dataset`, got={config.dataset}")


def get_callbacks(checkpoint_filepath):
    callbacks = []
    if FLAGS.model_dir:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="loss",
            save_freq=10
            # save_best_only=True
        )
        callbacks += [model_checkpoint_callback]

    if FLAGS.artifact_dir:
        log_dir = os.path.join(FLAGS.artifact_dir, "logs")
        callbacks += [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    if FLAGS.wandb:
        callbacks += [wandb.keras.WandbCallback()]

    return callbacks


def get_strategy():
    return tf.distribute.get_strategy()

def main(args):
    del args
    config = FLAGS.config

    if FLAGS.debug:
        tf.debugging.disable_traceback_filtering()

    try:
        os.makedirs(FLAGS.artifact_dir)
    except:
        pass

    strategy = get_strategy()

    if FLAGS.wandb:
        wandb.init(
            project=config.wandb_project_name,
            entity="reef-net",
            config=config.to_dict(),
        )

    autotune = tf.data.AUTOTUNE
    ds, dataset_size = load_reef_dataset(config, min_boxes_per_image=1)
    ds = ds.map(preprocess_data, num_parallel_calls=autotune)
    ds = ds.shuffle(config.batch_size * 2)
    ds = ds.repeat()
    ds = ds.padded_batch(
        config.batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )

    label_encoder = LabelEncoder()
    ds = ds.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    # input_shape = ds.element_spec[0].shape

    strategy = tf.distribute.MirroredStrategy()
    resnet50_backbone = get_backbone()
    # print(resnet50_backbone.summary())
    model = RetinaNet(config.num_classes, resnet50_backbone)

    
    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        # metrics=[keras_cv.metrics.MeanAveragePrecision()],
        run_eagerly=FLAGS.debug,
    )
    model.build((None, None, None, 3))
    model.summary()

    # datetime object containing current date and time
    # dd/mm/YY H:M:S
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = dt.replace("/", "_", -1)
    dt = dt.replace(":", "_", -1)
    dt = dt.replace(" ", "__", -1)

    checkpoint_filepath = os.path.abspath("./models/" + dt + "/model")
    print("Checkpoint filepath:", checkpoint_filepath)

    epochs = 100
    steps_per_epoch = dataset_size / (config.batch_size)
    if FLAGS.debug:
        steps_per_epoch = 3
    model.fit(
        ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=get_callbacks(checkpoint_filepath),
    )
    print("Fit Done")

    if FLAGS.model_dir:
        model.save(checkpoint_filepath)


if __name__ == "__main__":
    app.run(main)

import os
from datetime import datetime

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import wandb
from reef_net.loaders import load_reef_dataset, load_n_images
from reef_net.model import DecodePredictions
from reef_net.model import RetinaNet
from reef_net.model import get_backbone
from reef_net.preprocess import preprocess_data
from reef_net.preprocess import resize_and_pad_image
from reef_net.utils import LabelEncoder
from reef_net.utils import visualize_detections
from reef_net.callbacks import VisualizePredictions
import keras_cv

config_flags.DEFINE_config_file("config", "configs/main.py")

flags.DEFINE_bool("wandb", False, "Whether to run to wandb")
flags.DEFINE_bool("debug", False, "Whether or not to use extra debug utilities")
flags.DEFINE_string("artifact_dir", None, "Directory to store artifacts")
flags.DEFINE_string("model_dir", None, "Where to save the model after training")

FLAGS = flags.FLAGS


def get_callbacks(config, checkpoint_filepath, val_path, train_path):
    callbacks = []
    if FLAGS.model_dir:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_freq="epoch",
            save_format="tf",
            save_best_only=True,
        )
        callbacks += [model_checkpoint_callback]

    if FLAGS.artifact_dir:
        log_dir = os.path.join(FLAGS.artifact_dir, "logs")
        callbacks += [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
        val_image, val_labels, val_category = load_n_images(
            config, val_path, min_boxes_per_image=5, n=1
        )
        vis_callback = VisualizePredictions(
            val_image, val_labels, FLAGS.artifact_dir, "test"
        )
        train_image, train_labels, train_category = load_n_images(
            config, train_path, min_boxes_per_image=5, n=1
        )
        vis_callback_train = VisualizePredictions(
            val_image, val_labels, FLAGS.artifact_dir, "train"
        )
        callbacks += [vis_callback_train]

    if FLAGS.wandb:
        callbacks += [wandb.keras.WandbCallback()]

    return callbacks


def get_strategy():
    return tf.distribute.MirroredStrategy()


def get_checkpoint_path():
    # datetime object containing current date and time
    # dd/mm/YY H:M:S
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = dt.replace("/", "_", -1)
    dt = dt.replace(":", "_", -1)
    dt = dt.replace(" ", "__", -1)

    checkpoint_filepath = os.path.abspath("./models/" + dt + "/model")
    print("Checkpoint filepath:", checkpoint_filepath)

    return checkpoint_filepath


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
    print("Running with strategy:", str(strategy))
    print("Devices:", tf.config.list_physical_devices())

    if FLAGS.wandb:
        wandb.init(
            project=config.wandb_project_name,
            entity="reef-net",
            config=config.to_dict(),
        )

    autotune = tf.data.AUTOTUNE
    label_encoder = LabelEncoder()

    ########## ---------- XXXXXXXXXX ---------- ##########
    # This is for training data
    base_path = config.custom_path
    train_path = os.path.abspath(os.path.join(base_path, config.train_path))
    train_ds, train_dataset_size = load_reef_dataset(
        config, train_path, min_boxes_per_image=1
    )
    train_ds = train_ds.map(preprocess_data, num_parallel_calls=autotune)
    train_ds = train_ds.shuffle(config.batch_size * 8)
    train_ds = train_ds.repeat()
    train_ds = train_ds.padded_batch(
        config.batch_size, padding_values=(0.0, 1e-8, 1e-8, -1), drop_remainder=True
    )

    train_ds = train_ds.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.prefetch(autotune)  # This line is added

    ########## ---------- XXXXXXXXXX ---------- ##########
    # This is for validation dataset
    base_path = config.custom_path
    val_path = os.path.abspath(os.path.join(base_path, config.val_path))
    val_ds, val_dataset_size = load_reef_dataset(
        config, val_path, min_boxes_per_image=1
    )
    val_ds = val_ds.map(preprocess_data, num_parallel_calls=autotune)
    val_ds = val_ds.shuffle(config.batch_size * 8)
    val_ds = val_ds.repeat()
    val_ds = val_ds.padded_batch(
        config.batch_size, padding_values=(0.0, 1e-8, 1e-8, -1), drop_remainder=True
    )

    val_ds = val_ds.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    val_ds = val_ds.prefetch(autotune)  # This line is added

    checkpoint_filepath = get_checkpoint_path()

    strategy = tf.distribute.MirroredStrategy()
    resnet50_backbone = get_backbone()
    # print(resnet50_backbone.summary())
    model = RetinaNet(
        config.num_classes,
        resnet50_backbone,
        batch_size=config.batch_size,
    )

    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        # run_eagerly=FLAGS.debug,
        metrics=[
            keras_cv.metrics.COCOMeanAveragePrecision(
                class_ids=[1], bounding_box_format="xyxy"
            ),
            keras_cv.metrics.COCORecall(class_ids=[1], bounding_box_format="xyxy"),
        ],
    )
    model.build((None, None, None, 3))
    model.summary()

    epochs = 300
    steps_per_epoch = 1000  # train_dataset_size // (config.batch_size)
    validation_steps = 100

    if FLAGS.debug:
        epochs = 100
        steps_per_epoch = 1
        validation_steps = 1
    cbs = (get_callbacks(config, checkpoint_filepath, val_path, train_path),)
    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=cbs,
        verbose=1,
    )
    print("Fit Done")

    if FLAGS.model_dir:
        model.save(checkpoint_filepath)


if __name__ == "__main__":
    app.run(main)

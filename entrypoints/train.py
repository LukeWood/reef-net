import os
import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags
import tensorflow as tf

from reef_net.loaders import load_reef_dataset
from reef_net.model import get_backbone, RetinaNetLoss, RetinaNet, DecodePredictions
from reef_net.utils import LabelEncoder, visualize_detections
from reef_net.preprocess import preprocess_data, resize_and_pad_image


config_flags.DEFINE_config_file("config", "configs/main.py")

# config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "Whether to run to wandb")
flags.DEFINE_string("artifact_dir", None, "Directory to store artifacts")

FLAGS = flags.FLAGS

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
    # logging.info(f"Running with model_dir: {FLAGS.model_dir}")
    # logging.info(f"Running with artifact_dir: {FLAGS.artifact_dir}")

    if FLAGS.wandb:
        wandb.init(
            project=config.wandb_project_name, entity="reef-net", config=config.to_dict()
        )

    autotune = tf.data.AUTOTUNE
    ds = load_reef_dataset(config, min_boxes_per_image=1)
    ds = ds.shuffle(config.batch_size * 2)
    ds = ds.map(preprocess_data, num_parallel_calls=autotune)
    ds = ds.repeat()
    ds = ds.padded_batch(config.batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
    label_encoder = LabelEncoder()
    ds = ds.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.prefetch(autotune)
    # input_shape = ds.element_spec[0].shape

    resnet50_backbone = get_backbone()
    # print(resnet50_backbone.summary())
    loss_fn = RetinaNetLoss(1)
    model = RetinaNet(1, resnet50_backbone)

    optimizer = tf.optimizers.SGD(momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer, run_eagerly=True)
    model.build((None, None, None, 3))
    model.summary()

    for sample in ds.take(100):
        print("Hi", sample[0].shape)

    model.fit(
        ds.take(100),
        epochs=1
    )

if __name__ == "__main__":
    app.run(main)

# START BELOW HERE !!

# --- Setting up training parameters ---
# model_dir = "../reef_net/"
# label_encoder = LabelEncoder()
#
# num_classes = 80
# batch_size = 2
#
# learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
# learning_rate_boundaries = [125, 250, 500, 240000, 360000]
# learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
#     boundaries=learning_rate_boundaries, values=learning_rates
# )
#
# # --- Initializing and compiling model ---
# resnet50_backbone = get_backbone()
# loss_fn = RetinaNetLoss(num_classes)
# model = RetinaNet(num_classes, resnet50_backbone)
#
# optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
# model.compile(loss=loss_fn, optimizer=optimizer)
#
# # --- Setting up callbacks ---
# callbacks_list = [
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
#         monitor="loss",
#         save_best_only=False,
#         save_weights_only=True,
#         verbose=1,
#     )
# ]
#
#
# # --- Load the dataset ---
#
# ds = load_reef_dataset(config, min_boxes_per_image=0)
# # split into train/val sets
#
# # (train_dataset, val_dataset), dataset_info = tfds.load(
# #     "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
# # )
#
#
# # --- Setting up a tf.data pipeline ---
# autotune = tf.data.AUTOTUNE
# train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
# train_dataset = train_dataset.shuffle(8 * batch_size)
# train_dataset = train_dataset.padded_batch(
#     batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
# )
# train_dataset = train_dataset.map(
#     label_encoder.encode_batch, num_parallel_calls=autotune
# )
# train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
# train_dataset = train_dataset.prefetch(autotune)
#
# val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
# val_dataset = val_dataset.padded_batch(
#     batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
# )
# val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
# val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
# val_dataset = val_dataset.prefetch(autotune)
#
#
# # --- Training the model ---
# # Uncomment the following lines, when training on full dataset
# # train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# # val_steps_per_epoch = \
# #     dataset_info.splits["validation"].num_examples // batch_size
#
# # train_steps = 4 * 100000
# # epochs = train_steps // train_steps_per_epoch
#
# epochs = 1
#
# # Running 100 training and 50 validation steps,
# # remove `.take` when training on the full dataset
#
# model.fit(
#     train_dataset.take(100),
#     validation_data=val_dataset.take(50),
#     epochs=epochs,
#     callbacks=callbacks_list,
#     verbose=1,
# )
#
#
# # --- Loading weights ---
# # Change this to `model_dir` when not using the downloaded weights
# weights_dir = "data"
#
# latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
# model.load_weights(latest_checkpoint)
#
#
# # --- Building inference model
# image = tf.keras.Input(shape=[None, None, 3], name="image")
# predictions = model(image, training=False)
# detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
# inference_model = tf.keras.Model(inputs=image, outputs=detections)
#
#
# # --- Generating detections ---
# def prepare_image(image):
#     image, _, ratio = resize_and_pad_image(image, jitter=None)
#     image = tf.keras.applications.resnet.preprocess_input(image)
#     return tf.expand_dims(image, axis=0), ratio
#
# # REPLACE WITH OUR DATASET
# val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
# int2str = dataset_info.features["objects"]["label"].int2str
#
# for sample in val_dataset.take(2):
#     image = tf.cast(sample["image"], dtype=tf.float32)
#     input_image, ratio = prepare_image(image)
#     detections = inference_model.predict(input_image)
#     num_detections = detections.valid_detections[0]
#     class_names = [
#         int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
#     ]
#     visualize_detections(
#         image,
#         detections.nmsed_boxes[0][:num_detections] / ratio,
#         class_names,
#         detections.nmsed_scores[0][:num_detections],
#     )
#
#
#
#

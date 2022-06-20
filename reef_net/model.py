import keras_cv
import numpy as np
import tensorflow as tf
from absl import flags
from ml_collections.config_flags import config_flags
from tensorflow import keras

from reef_net import layers as layers_lib
from reef_net import losses as losses_lib
from reef_net.loaders import load_reef_dataset
from reef_net.utils import AnchorBox
from reef_net.utils import convert_to_corners


# --- Building RetinaNet using a subclassed model ---
class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
    """

    def __init__(
        self,
        num_classes,
        batch_size,
        backbone=None,
        alpha=0.25,
        gamma=2.0,
        delta=1.0,
        name="RetinaNet",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.fpn = layers_lib.FeaturePyramid(backbone)
        self.num_classes = num_classes
        self.batch_size = batch_size

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.classification_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

        self._loss = losses_lib.RetinaNetLoss(
            num_classes=num_classes,
            classification_loss=losses_lib.RetinaNetClassificationLoss(alpha, gamma),
            box_loss=losses_lib.RetinaNetBoxLoss(delta),
        )

        self.decoder = layers_lib.DecodePredictions(
            num_classes=num_classes, batch_size=batch_size
        )

        self.gradient_norm_metric = tf.keras.metrics.Mean(name="gradient_norm")

    def call(self, x, training=False):
        features = self.fpn(x, training=training)
        N = tf.shape(x)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.classification_head(feature), [N, -1, self.num_classes])
            )

        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        train_preds = tf.concat([box_outputs, cls_outputs], axis=-1)

        decoded = self.decoder(x, train_preds)
        result = self._encode_to_ragged(decoded)
        pred_for_inference = result.to_tensor(default_value=-1)

        return {"train_preds": train_preds, "inference": pred_for_inference}

    def _update_metrics(self, y_for_metrics, result):
        # COCO metrics are all stored in compiled_metrics
        # This tf.cond is needed to work around a TensorFlow edge case in Ragged Tensors
        tf.cond(
            tf.shape(result)[2] != 0,
            lambda: self.compiled_metrics.update_state(y_for_metrics, result),
            lambda: None,
        )

    def _metrics_result(self, loss):
        metrics_result = {m.name: m.result() for m in self.metrics}
        metrics_result["loss"] = loss
        return metrics_result

    def train_step(self, data, training=True):
        x, (y_true, y_for_metrics) = data
        x = tf.cast(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            predictions = self(x, training=training)
            loss = self._loss(y_true, predictions["train_preds"])
            for extra_loss in self.losses:
                loss += extra_loss

        self._update_metrics(y_for_metrics, predictions["inference"])

        # Training specific code
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # clip grads to prevent explosion
        gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
        self.gradient_norm_metric.update_state(gradient_norm)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return metric result

        return self._metrics_result(loss)

    def test_step(self, data):
        x, (y_true, y_for_metrics) = data
        x = tf.cast(x, dtype=tf.float32)

        predictions = self(x, training=False)
        loss = self._loss(y_true, predictions["train_preds"])
        for extra_loss in self.losses:
            loss += extra_loss

        self._update_metrics(y_for_metrics, predictions["inference"])

        return self._metrics_result(loss)

    def inference(self, x):
        predictions = self.predict(x)
        return predictions["inference"]


# --- Building the ResNet50 backbone ---
def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

    @property
    def metrics(self):
        return super().metrics + [self.gradient_norm_metric]


# --- Building the classification and box regression heads. ---
def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head

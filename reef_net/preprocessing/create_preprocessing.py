import tensorflow as tf

from reef_net.preprocessing.random_flip_horizontal import random_flip_horizontal
from reef_net.preprocessing.resize_and_pad_image import resize_and_pad_image
from reef_net.utils import convert_to_corners
from reef_net.utils import convert_to_xywh
from reef_net.utils import swap_xy


def noop(image, bounding_boxes):
    return image, bounding_boxes


def apply_random_flip(image, bounding_boxes):
    image, bounding_boxes = random_flip_horizontal(image, bounding_boxes)
    return image, bounding_boxes


def get_augmentation_function(augmentation_mode):
    if augmentation_mode is None:
        return noop
    if augmentation_mode == "random_flip":
        return apply_random_flip
    else:
        raise ValueError(
            f"Unsupported augmentation_mode, {augmentation_mode}, "
            "was set in the training config.  config.augmentation_mode should be set to "
            "one of [None, 'random_flip']"
        )


def create_preprocessing_function(augmentation_mode):
    """creates a preprocessing function for use with bounding boxes and images.

    Arguments:
        sample: A dict representing a single training sample.

    Returns:
        a sample-wise preprocessing function that takes the arguments:
            image: Resized and padded image with random horizontal flipping applied.
            bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
                of the format `[x, y, width, height]`.
            class_id: An tensor representing the class id of the objects, having
                shape `(num_objects,)`.
    """

    augmentation_function = get_augmentation_function(augmentation_mode)

    def preprocess(image, bounding_boxes, class_ids):
        bounding_boxes = swap_xy(bounding_boxes)
        image, bounding_boxes = augmentation_function(image, bounding_boxes)
        image, image_shape, _ = resize_and_pad_image(image)
        corners_boxes = tf.stack(
            [
                bounding_boxes[:, 0] * image_shape[1],
                bounding_boxes[:, 1] * image_shape[0],
                bounding_boxes[:, 2] * image_shape[1],
                bounding_boxes[:, 3] * image_shape[0],
            ],
            axis=-1,
        )
        xywh_boxes = convert_to_xywh(corners_boxes)
        return image, xywh_boxes, corners_boxes, class_ids

    return preprocess

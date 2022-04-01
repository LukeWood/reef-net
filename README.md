# Coral Reef Object Detection

Dataset: https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/overview

Let's work with prototypes in the `prototypes` folder and follow the following
architecture for the final project.


### Some Helpful links:

- https://www.tensorflow.org/guide/data
- [RetinaNet in Keras](https://keras.io/examples/vision/retinanet)
- https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
- https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/
- [Another sample Keras object detection model (Yolov3) - very old code](https://github.com/8000net/YOLOv3)

# Project Structure
The project should be structured as follows:

- the `data_loader` module loads the data.
- `preprocess` augments the data.
- `model` contains the modeling code.
- `visualize` module to perform visualization.
- a main driver `train.py` puts it all together to train the model.

### Data Loader

The data_loader should take the `.csv` file distributed with the Kaggle dataset and load
a dataset into a `tf.data.Dataset` object.  This data pipeline tends to be pretty nice
to work with and very easy to distribute/parallelize which will save us time in the
long run.

Harsh can give this a try.

### Preprocess

We can use the `albumnations` library to augment images with bounding boxes.

Recommended reading:

- https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

###

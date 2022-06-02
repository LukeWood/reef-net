![reefnet logo](media/reefnet.png)

<img src="media/demo_image.png" width="400px"/>

<br/>

ReefNet is a RetinaNet implementation written in pure Keras developed for the
[Crown-of-Thorns Great Barrier Reef Dataset](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/overview)
The project includes a custom `tf.data.Dataset` loader, a `keras.Model` subclass
implementation of RetinaNet, integration with the `keras_cv.metrics.COCOMeanAveragePrecision`
and `keras_cv.metrics.COCORecall` metrics, and a Keras callback to visualize
predictions.

## Quickstart

To get up in and running, you should run:
```bash
python setup.py develop
```

Next, you will need to download the dataset:
```
kaggle competitions download -c tensorflow-great-barrier-reef
mkdir tensorflow-great-barrier-reef
mv tensorflow-great-barrier-reef.zip tensorflow-great-barrier-reef
cd tensorflow-great-barrier-reef
unzip tensorflow-great-barrier-reef.zip
```

To test that you are properly setup, try running:

```
python entrypoints/show_samples.py
```

This script should show two images: one with no annotations and one with annotations.

## Training

To train, first follow the "Quickstart" section.

After following quickstart, you should be able to run the following:

```bash
python entrypoints/train.py --artifact_dir=artifacts
```

## Future Efforts

Currently, the model does not achieve strong performance.  The following steps should be taken to
improve the model performance:

- experiment with multiple AnchorBox configurations
- experiment with transfer learning
- include image augmentation techniques

## Contributing

If you'd like to contribute, feel free to open a PR improving the cleanliness of the codebase,
experimenting with new anchorbox configurations, or including more data augmentation techniques.

## Thanks!

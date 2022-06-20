#!/bin/bash
mkdir -p data/tensorflow-great-barrier-reef
kaggle competitions download -c tensorflow-great-barrier-reef
mv tensorflow-great-barrier-reef.zip data/tensorflow-great-barrier-reef
cd data/tensorflow-great-barrier-reef
unzip tensorflow-great-barrier-reef.zip

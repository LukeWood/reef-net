#!/bin/sh

set -x
rm -rf make_video/
mkdir make_video
cp $1/* make_video/
python shell/pad-image-names.py
rm make_video/ground_truth.png
name=${2:-learning}
ffmpeg -framerate 1 -pattern_type glob -i "make_video/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p media/$name.mp4

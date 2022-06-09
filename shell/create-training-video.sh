#!/bin/sh

set -x
rm -rf make_video/
mkdir make_video
cp $1/* make_video/
python shell/pad-image-names.py
rm make_video/ground_truth.png
name=${2:-learning}
ffmpeg -framerate 1 -i "make_video/%03d.png" -vcodec libx264 \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" media/$name.mp4

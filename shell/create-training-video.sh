#!/bin/sh

set -x
rm -rf make_video/
mkdir make_video
cp $1/* make_video/
rm make_video/ground_truth.png
python3 shell/pad-image-names.py
name=${2:-learning}

RESIZE_FILTER="pad=ceil(iw/2)*2:ceil(ih/2)*2"
FRAME_FILTER="drawtext=fontfile=Arial.ttf: text='Epoch %{frame_num}': start_number=1: x=10: y=10: fontcolor=black: fontsize=20:"
ffmpeg -f image2 -framerate 1 -pattern_type glob -i "make_video/*.png" -start_number 1 -vf "$RESIZE_FILTER,$FRAME_FILTER" -pix_fmt yuv420p media/$name.mp4
ffmpeg -framerate 1 -pattern_type glob -i "make_video/*.png" -vf "$FRAME_FILTER,$RESIZE_FILTER" media/$name.gif

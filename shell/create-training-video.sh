bin/sh

set -x
rm -rf make_video/
mkdir make_video
cp $1/* make_video/
rm make_video/ground_truth.png
name=${2:-learning}
ffmpeg -framerate 1 -pattern_type glob -i "make_video/*.png" media/$name.mp4

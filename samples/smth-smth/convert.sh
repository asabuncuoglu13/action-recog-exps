#!/bin/bash

FILES="/datasets/20bn_something_something/v2/videos/*"
OUT="~/smth"
i=0
for f in $FILES; do
  i=$((i+1))
  if ((i % 10000)); then
    ffmpeg -y -i $f -filter:v fps=4 -c:v libvpx-vp9 -s 144x81 -b:v 1M -strict -2 -movflags faststart -acodec libvorbis "$OUT$(basename $f)"
  fi
done

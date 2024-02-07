ffmpeg -y -f image2 -r 20 -pattern_type glob -i 'frames/river_????.png' -crf 22 video.mp4

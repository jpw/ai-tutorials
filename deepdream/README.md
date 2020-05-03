# deepdream

from: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/deepdream.ipynb

## setup

 - run `py3env-tensorflow-create.sh`: sets up a python3 env with tensorflow
 - `source env/bin/activate` to activate the env
 
 Then, if `which python` is pointing at the interpreter in your env, `./dreamer.py`
 
 ## bonus points - video!

MP4:
 `ffmpeg -f image2 -r 1 -pattern_type glob -i 'renders/reference/10-combinations/mixed*.png' -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" -vcodec mpeg4 -y ./dog.mp4`

H.264 much better quality:
`ffmpeg -f image2 -r 1 -pattern_type glob -i 'renders/reference/10-combinations/mixed*.png' -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" -crf 18 -vcodec libx264 -y ./dog-264.h264`

Still experimenting with interpolation, but pretty good:
 `ffmpeg -i dog-264.h264 -vf minterpolate=50,tblend=all_mode=average -crf 18 -vcodec libx264 minterpolate-fancy.mp4`

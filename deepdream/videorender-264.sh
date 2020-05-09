ffmpeg -f image2 -r 1 -pattern_type glob -i 'renders/*.png' -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" -crf 18 -vcodec libx264 -y ./tmp.h264
ffmpeg -i tmp.h264 -vf minterpolate=50,tblend=all_mode=average -crf 18 -vcodec libx264 video-interpolated.h264


# render a H.264 video from the still images
ffmpeg -f image2 -r 1 -pattern_type glob -i 'renders/*.png' -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" -crf 18 -vcodec libx264 -y ./tmp.h264
# interpolate frames and output in an MP4 wrapper (but still 264 codec). faststart helps browsers and YT start quicker
ffmpeg -i tmp.h264 -movflags +faststart -vf minterpolate=50,tblend=all_mode=average -crf 18 -f mp4 video-interpolated.mp4


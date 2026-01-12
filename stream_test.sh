#!/usr/bin/env bash
ffmpeg  \
-rtsp_transport tcp -i rtsp://192.168.1.78:8080/h264.sdp \
-c:v libx264  \
-preset veryfast    \
-crf 37 \
-f flv  rtmps://dc4-1.rtmp.t.me/s/3234968082:IrMYQ3O_BVrXDH_2G5Djag
#-stream_loop -1 \
#-i data/birdclip_20251214_114345.mp4 \
#-g 40 -keyint_min 40 \
#-vf fps=20 \
#-f flv -
#-f null -
#-b:v 1200k -maxrate 1200k -bufsize 4800k \
#-c:v libx264   \
#-c:a aac -b:a 96k \
#-pix_fmt yuv420p \
#-b:v 2500k -maxrate 2500k -bufsize 5000k \
#-c:a aac -b:a 128k \


#ffmpeg -stream_loop -1 -i data/birdclip_20251214_114345.mp4 -c:v libx264 -preset ultrafast -vf fps=20 -tune zerolatency -profile:v baseline -f null -
import vlc

print ( vlc.libvlc_get_version())

p = vlc.MediaPlayer('rtsp://192.168.1.30:8554/')
#p.play()

print ( p)


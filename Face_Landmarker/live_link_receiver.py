from __future__ import annotations
import socket

from pylivelinkface import PyLiveLinkFace, FaceBlendShape

UDP_PORT = 11111
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try: 
    # open a UDP socket on all available interfaces with the given port
    s.bind(("", UDP_PORT)) 
    while True: 
        data, addr = s.recvfrom(1024) 

        success, live_link_face = PyLiveLinkFace.decode(data)
        if success:
            # get the blendshape value for the HeadPitch and print it
            i = 0 
            while (i < 61):
                pitch= live_link_face.get_blendshape(FaceBlendShape(i))
                print (FaceBlendShape(i), pitch)     
                i += 1     
            pass

except KeyboardInterrupt:
    pass
       
finally: 
    s.close()








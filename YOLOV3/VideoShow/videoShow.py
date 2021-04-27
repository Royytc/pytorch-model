import time
import cv2
import numpy as np
from PIL import Image


capture=cv2.VideoCapture(0)
while True:
    ref,frame=capture.read()
    #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=Image.fromarray(np.uint8(frame))
    frame=np.array(frame)
    image_shape = np.array(np.shape(frame)[0:2])
    print(image_shape)
    cv2.imshow("video",frame)
    c = cv2.waitKey(1) & 0xff
    if c == 27:
        capture.release()
        break
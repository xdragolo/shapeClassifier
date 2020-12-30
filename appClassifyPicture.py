import os
from glob import glob
import numpy as np
import cv2
import keras
from matplotlib import pyplot as plt

from imagePreprocessing import getData, resizePic

PATH = os.path.abspath(os.path.join('.','Pictures'))
madeImages = glob(os.path.join(PATH, '*.png'))


cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

img_counter = 0

print('Make photo with SPACE and close window with ESC')

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #  higher contrast
        alpha = 1.5
        beta = 0
        adjusted = cv2.convertScaleAbs(grayFrame, alpha=alpha, beta=beta)
        img_name = "./Pictures/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, adjusted)
        print("{} written!".format(img_name))
        img_counter += 1
cam.release()
cv2.destroyAllWindows()

model = keras.models.load_model('./model')
X = resizePic(10,10, madeImages)
y = model.predict(X)

def getLabel(y):
    if np.round(y[0]) == 1:
        return 'circle'
    if np.round(y[0]) == 0:
        return 'star'

for i in range(len(madeImages)):
    print(madeImages[i], ' is ', getLabel(y[i]))

# img =cv2.imread('Pictures/opencv_frame_0.png',0)
# pic = cv2.resize(img,(10,10), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('pic',pic)
# cv2.waitKey(0)

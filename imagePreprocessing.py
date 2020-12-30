import os
import random
from glob import glob
import matplotlib.pylab as plt
import cv2
import numpy as np

PATH = os.path.abspath(os.path.join('.','archive'))
CIRCLES_BASE = os.path.join(PATH,'shapes','circle')
STAR_BASE = os.path.join(PATH,'shapes','star')

circlesImage = glob(os.path.join(CIRCLES_BASE, '*.png'))
starImages = glob(os.path.join(STAR_BASE, '*.png'))

def resizePic(width = 10, height =10,images = [],noise=False):
    X = []
    for img in images:
        fullSizeImage = cv2.imread(img)
        pic = cv2.resize(fullSizeImage,(width,height), interpolation=cv2.INTER_CUBIC)
        pic = np.concatenate(pic).reshape(300)
        if noise:
        # data are too perfect, adding some noise
            pic = np.round(pic * np.random.rand(300))
        X.append(pic)
    return np.array(X)



def getData(width = 10, height = 10):
    circles = resizePic(width, height, circlesImage)
    # np.save('circles', circles)
    stars = resizePic(width, height, starImages)
    stars = stars[:len(circles)]
    # np.save('stars', stars)
    return circles, stars



# width = 10
# height = 10
# circles = resizePic(width, height, circlesImage)
# print(len(circles),len(circles[0]))
# circles = np.array(circles)
#
# np.save('circles', circles)
# stars = resizePic(width, height, starImages)
# stars = np.array(stars)
# stars = stars[:len(circles)]
# np.save('stars', stars)
# print(stars)








#  Fetch all the image file names using glob
#  Read all the images using cv2.imread()
#  Store all the images into a list
#  Create a VideoWriter object using cv2.VideoWriter()
#  Save the images to video file using cv2.VideoWriter().write()
#  Release the VideoWriter and destroy all windows.

import cv2
import numpy as np
import glob
import os

path = '/home/arnold/raindrop-detection-cnn/mesonet/'
img_array = []
for count in range(len(os.listdir(path))):
#for filename in glob.glob('C:/New folder/Images/*.jpg'):
#     filename = './path/mask_frame_' + str(count) + '.jpg'
    filename = path + str(count) + '.jpg'
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

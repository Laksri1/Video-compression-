#
# EE596 Mini Project 
# E/18/023
#

import cv2
import numpy as np



original_img = cv2.imread("image.jpg")
original_img_array = np.array(original_img)

cv2.imshow('origional IMAGE (E/18/023)',original_img_array)
cv2.waitKey(0) 
cv2.destroyAllWindows()

gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
gray_img_array = np.array(gray_img)

cv2.imshow('origional IMAGE (E/18/023)',gray_img_array)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# Define macro block size
windowsize_r = 5
windowsize_c = 5


arr = np.split(gray_img_array, 5)
arr = np.array([np.split(x, 5, 1) for x in arr])

print(arr[0][0])

cv2.imshow('origional IMAGE (E/18/023)',arr[4][4])
cv2.waitKey(0) 
cv2.destroyAllWindows()



# Crop out the window and calculate the histogram
# for r in range(0,gray_img_array.shape[0] - windowsize_r, windowsize_r):
#     for c in range(0,gray_img_array.shape[1] - windowsize_c, windowsize_c):
#         window = gray_img_array[r:r+windowsize_r,c:c+windowsize_c]
#         hist = np.histogram(window,bins=256)

# print(hist)

# for h1, h2 in hist:
#     print(np.all(h1[0] == h2[0]))
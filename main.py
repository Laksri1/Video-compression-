#
# EE596 Mini Project 
# E/18/023
#

import cv2
import numpy as np
import math
from scipy.fftpack import dct


# DCT ========================================================
def dct(block):
    return np.fft.fft2(block, norm="ortho")


# Quantize ===================================================
def quantize(block, quantization_matrix):
    return np.round(block / quantization_matrix)


# Dequantize =================================================
def dequantize(block, quantization_matrix):
    return block * quantization_matrix

# Inverse DCT ================================================
def idct(block):
    return np.fft.ifft2(block, norm="ortho")


# Quantizer ===================================================
def quantiser(array):

    # max=np.max(red_array)
    # min=np.min(red_array)
    max=255
    min=0

    # q = (max-min)/7

    q=36.4285714

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j]>=min and array[i][j]<min+q/2 :
                    array[i][j]= min
            elif array[i][j]>=min+q/2 and array[i][j]<min+3*q/2 :
                    array[i][j]= min + q
            elif array[i][j]>=min+3*q/2 and array[i][j]<min+5*q/2 :
                    array[i][j]= min + 2*q
            elif array[i][j]>=min+5*q/2 and array[i][j]<min+7*q/2 :
                    array[i][j]= min + 3*q
            elif array[i][j]>=min+7*q/2 and array[i][j]<min+9*q/2 :
                    array[i][j]= min + 4*q
            elif array[i][j]>=min+9*q/2 and array[i][j]<min+11*q/2 :
                    array[i][j]= min + 5*q
            elif array[i][j]>=min+11*q/2 and array[i][j]<min+13*q/2 :
                    array[i][j]= min + 6*q
            elif array[i][j]>=min+13*q/2 and array[i][j]<min+15*q/2 :
                    array[i][j]= min + 7*q

    return np.array(array)

# Save file ===================================================
def save_dict_to_txt(dictionary, file_path):
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

# Run Lenghth =================================================
def run_length_coding(matrix):

    rle_list = []
    current_run = None
    run_length = 0

    for row in matrix:
        for element in row:
            if current_run is None:
                current_run = element
                run_length = 1
            elif current_run == element:
                run_length += 1
            else:
                rle_list.append((current_run, run_length))
                current_run = element
                run_length = 1

    # Add the last run
    rle_list.append((current_run, run_length))
    print(rle_list)
    return rle_list


# Run Length Decode ===========================================
def run_length_decode(rle_list):
    decoded_matrix = []

    for element, run_length in rle_list:
        decoded_matrix.extend([element] * run_length)

    return decoded_matrix


# Main ========================================================
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
windowsize_r = 60 #use 60 for 8x8 matrix
windowsize_c = 60

arr = np.split(gray_img_array, windowsize_r)
arr = np.array([np.split(x, windowsize_c, 1) for x in arr])

imageAfter_DCT =np.array(arr)
imageAfter_Quantize =np.array(arr)
imageAfter_Coding ={}

quantized_matrix=[
    [3, 5, 7, 9, 11, 13, 15, 17],
    [5, 7, 9, 11, 13, 15, 17, 19],
    [7, 9, 11, 13, 15, 17, 19, 21],
    [9, 11, 13, 15, 17, 19, 21, 23],
    [11, 13, 15, 17, 19, 21, 23, 25],
    [13, 15, 17, 19, 21, 23, 25, 27],
    [15, 17, 19, 21, 23, 25, 27, 29],
    [17, 19, 21, 23, 25, 27, 29, 31]
]

for x in range(windowsize_r):
    for y in range(windowsize_c):
        imageAfter_DCT[x][y] = dct(arr[x][y])

for x in range(windowsize_r):
    for y in range(windowsize_c):
        imageAfter_Quantize[x][y] = quantize(imageAfter_DCT[x][y],quantized_matrix)

for x in range(windowsize_r):
    for y in range(windowsize_c):
        imageAfter_Coding[str(x)+"_"+str(y)] = run_length_coding(imageAfter_Quantize[x][y])

save_dict_to_txt(imageAfter_Coding,"encode.txt")

print(arr[0][0])
print(dct(arr[0][0]))
print(idct(dct(arr[0][0])).real)

cv2.imshow('origional IMAGE (E/18/023)',arr[4][4])
cv2.waitKey(0) 
cv2.destroyAllWindows()

cv2.imshow('origional IMAGE (E/18/023)',(idct(dct(arr[4][4]))).astype(np.uint8))
cv2.waitKey(0) 
cv2.destroyAllWindows()

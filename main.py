#
# EE596 Mini Project 
# E/18/023
#

import cv2
import numpy as np


# DCT ========================================================

def dct(block):
    return np.fft.fft2(block, norm="ortho")


# Inverse DCT ================================================

def idct(block):
    return np.fft.ifft2(block, norm="ortho")


# Probability =================================================

def probability(q_vals):
  
    unique_values, counts = np.unique(q_vals, return_counts=True)
    
    value_counts = dict(zip(unique_values, counts))

    for key in value_counts:
        value_counts[key]=value_counts[key]/(16*16)

    return value_counts


# Quantizer =================================================

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


# Sort data ========================================================

def sort(input_dict):
    items = list(input_dict.items())
    n = len(items)

    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if items[j][1] > items[j + 1][1]:
                items[j], items[j + 1] = items[j + 1], items[j]

    sorted_img = dict(items)
    
    return sorted_img


# Huffman ===========================================================

def huffman(sorted_dict,input_dict):
    for key in input_dict:
        input_dict[key] = ""

    while (len(sorted_dict)>1):
        new_value = list(sorted_dict.values())[0] + list(sorted_dict.values())[1]
        key = str(list(sorted_dict.keys())[0])+"_"+str(list(sorted_dict.keys())[1])

        sorted_dict[key] = new_value

        result = str(list(sorted_dict.keys())[0]).split('_')

        for i in result:
            input_dict[int(i)] = "0" + input_dict[int(i)]

        sorted_dict.pop(list(sorted_dict.keys())[0])

        result = str(list(sorted_dict.keys())[0]).split('_')

        for i in result:
            input_dict[int(i)] = "1"+input_dict[int(i)]

        sorted_dict.pop(list(sorted_dict.keys())[0])

        # Step 4: Sort the dictionary again using bubblesort
        sorted_dict = sort(sorted_dict)

    return input_dict



# Main =======================================================
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
windowsize_r = 5 #use 60 for 8x8 matrix
windowsize_c = 5

arr = np.split(gray_img_array, windowsize_r)
arr = np.array([np.split(x, windowsize_c, 1) for x in arr])

imageAfter_DCT =np.array(arr)
imageAfter_Quantize =np.array(arr)
img_probability =np.array(arr)
sorted_img_prob =np.array(arr)
huf_img =np.array(arr)

for x in range(windowsize_r):
    for y in range(windowsize_c):
        imageAfter_DCT[x][y] = dct(arr[x][y])

for x in range(windowsize_r):
    for y in range(windowsize_c):
        imageAfter_Quantize[x][y] = quantiser(imageAfter_DCT[x][y])

for x in range(windowsize_r):
    for y in range(windowsize_c):
        img_probability[x][y] = probability(imageAfter_Quantize[x][y])

codeBook = img_probability.copy()

for x in range(windowsize_r):
    for y in range(windowsize_c):
        sorted_img_prob[x][y] = sort(img_probability[x][y])

for x in range(windowsize_r):
    for y in range(windowsize_c):
        huf_img[x][y] = huffman(sorted_img_prob[x][y], codeBook[x][y])



print(arr[0][0])
print(dct(arr[0][0]))
print(idct(dct(arr[0][0])).real)

cv2.imshow('origional IMAGE (E/18/023)',arr[4][4])
cv2.waitKey(0) 
cv2.destroyAllWindows()

cv2.imshow('origional IMAGE (E/18/023)',(idct(dct(arr[4][4]))).astype(np.uint8))
cv2.waitKey(0) 
cv2.destroyAllWindows()

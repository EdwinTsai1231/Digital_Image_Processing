"""
CH2
	1. 將Lena512放大兩倍
		a)使用Nearest neighbor interpolation
		b)Bilinear interpolation
	2. 將Lena512影像上下翻轉
"""
import math
import numpy as np
import cv2 as cv

def UpSideDown (old_image):
    height,width,c = old_image.shape
    new_image = np.zeros((height,width,c))
    for i in range(height):
        for j in range(width):
             new_image[i,j,:] = image[height-1-i,j,:]

    return new_image.astype(np.uint8)

def nearestNeighbor (old_image , new_h , new_w):
    old_h , old_w, c = old_image.shape
    resized_image = np.zeros((new_h,new_w,c))
    enlarge = int(math.sqrt((new_h * new_w) / (old_h * old_w)))
    for i in range(new_h):
        for j in range(new_w):
            x = min(old_h-1 , math.floor(i/enlarge))
            y = min(old_w-1 ,math.floor(j/enlarge))

            resized_image[i,j,:] = old_image[x,y,:]

    return resized_image.astype(np.uint8)


def bilinearInterpolation (old_image , new_h , new_w):
    old_h , old_w , c = old_image.shape
    resized_image = np.zeros((new_h,new_w,c))
    w_scale = old_w / new_w if new_w!=0 else 0
    h_scale = old_h / new_h if new_h != 0 else 0
    for i in range(new_h):
        for j in range(new_w):
            x = i*h_scale ;
            y = j*w_scale ;
            x_floor = math.floor(x);
            x_ceil = min(old_h-1 , math.ceil(x))
            y_floor = math.floor(y);
            y_ceil = min(old_w - 1, math.ceil(y))

            if x_floor==x_ceil and y_floor==y_ceil :
                pixel = old_image[int(x) , int(y),:]
            elif x_floor==x_ceil:
                pixel1 = old_image[int(x),int(y_floor),:]
                pixel2 = old_image[int(x),int(y_ceil),:]
                pixel = pixel1 * (y_ceil-y) + pixel2 * (y-y_floor)
            elif y_floor==y_ceil:
                pixel1 = old_image[int(x_floor), int(y), :]
                pixel2 = old_image[int(x_ceil), int(y), :]
                pixel = pixel1 * (x_ceil - x) + pixel2 * (x-x_floor)
            else:
                p1 = old_image[int(x_floor) , int(y_floor),:]
                p2 = old_image[int(x_floor), int(y_ceil), :]
                p3 = old_image[int(x_ceil), int(y_floor), :]
                p4 = old_image[int(x_ceil), int(y_ceil), :]
                pixel1 = p1 * (x_ceil - x) + p3 * (x - x_floor)
                pixel2 = p2 * (x_ceil - x) + p4 * (x - x_floor)
                pixel = pixel1 * (y_ceil-y) + pixel2 * (y-y_floor)

            resized_image[i,j,:] = pixel

    return resized_image.astype(np.uint8)

image = cv.imread('input/cat.jpeg')
cv.imshow('original', image)
x, y, _ = image.shape

"""
NearestNeighbor_image = cv.resize(image, (x*2, y*2), interpolation=cv.INTER_NEAREST)
cv.imshow('Nearest neighbor interpolation', NearestNeighbor_image)
cv.imwrite('output/Nearest_neighbor_interpolation.bmp', NearestNeighbor_image)
"""

NearestNeighbor_image = nearestNeighbor(image,2048,2048)
cv.imshow('Nearest neighbor interpolation', NearestNeighbor_image)
cv.imwrite('output/Nearest_neighbor_interpolation_cat.jpeg', NearestNeighbor_image)

Bilinear_image = bilinearInterpolation(image , 2048 , 2048)
cv.imshow('Bilinear interpolation', Bilinear_image)
cv.imwrite('output/Bilinear_image_interpolation_cat.jpeg', Bilinear_image)

upsidedown_image = UpSideDown(image)
cv.imshow('Up Side Down', upsidedown_image)
cv.imwrite('output/upsidedown_image_cat.jpeg', upsidedown_image)

"""
upsidedown_image = cv.flip(image, 0)
cv.imshow('Up Side Down', upsidedown_image)
cv.imwrite('output/upsidedown_image.bmp', upsidedown_image)
"""

cv.waitKey(0)
cv.destroyAllWindows()

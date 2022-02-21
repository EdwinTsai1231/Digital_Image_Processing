import cv2
import numpy
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def reconstruct(image1 , image2 , image3):
    temp = cv.bitwise_or(image1 , image2)
    result = cv.bitwise_or(temp , image3)
    return result

def bitplane(image , plane):
    for i in range(8):
        x = 2 ** i
        plane[:, :, i] = cv.bitwise_and(image, x)
        temp = plane[:, :, i] > 0
        temp_plane = np.copy(plane)
        temp_plane[temp] = 255
        cv.imshow(str(i), temp_plane[:, :, i])
        cv.imwrite('output/' + str(i) + '.png', plane[:, :, i])

def laplacian_filter2(image) :
    m, n = image.shape
    filter = np.zeros((m, n, 2), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            filter[i][j] = -((i - m / 2) ** 2 + (j - n / 2) ** 2)

    ft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    ft_shifted = np.fft.fftshift(ft)

    applied = ft_shifted * filter
    f_ishift = np.fft.ifftshift(applied)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back

def laplacian_filter1(image):
    laplician = np.array(([1,1,1],[1,-8,1],[1,1,1]), dtype=float)
    w,h = image.shape
    pad_image = np.zeros((w+2 , h+2))
    pad_image[1:-1 , 1:-1] = image
    new_image = np.zeros(image.shape)
    for i in range(w) :
        for j in range(h):
            temp = np.array(pad_image[i:i+3 , j:j+3])
            result = np.dot(temp , laplician)
            new_image[i,j] = sum(sum(result))
    return new_image

image = cv.imread('input/100-dollars.tif' , 0)
height, width = image.shape
plane = np.zeros((height,width,8) , dtype=numpy.uint8)
mask = np.zeros((height , width) ,dtype=numpy.uint8)
cv.imshow("original" , image)
cv.imwrite("output/original_100-dollars.png" , image)

bitplane(image, plane)

pic_reconstruct = reconstruct(plane[:,:,7] , plane[:,:,6] , plane[:,:,5])
cv.imshow("reconstruct", pic_reconstruct)
cv.imwrite("output/reconstruct.png" , pic_reconstruct)

image2 = cv.imread('input/blurry_moon.tif')
cv.imwrite('output/original_moon.png' , image2)
image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
lap_image2 = laplacian_filter2(image2)
lap_image1 = laplacian_filter1(image2)
imgplot2 = plt.imshow(lap_image2, cmap="gray")
imgplot1 = plt.imshow(lap_image1, cmap='gray')
plt.show()
plt.imsave('output/laplacian1.png',lap_image1,cmap="gray")
plt.imsave('output/laplacian2.png',lap_image2,cmap="gray")

cv.waitKey(0)
cv.destroyAllWindows()
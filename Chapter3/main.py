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
        cv.imwrite('output/' + str(i) + '.tif', plane[:, :, i])

def laplacian_filter(image) :
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



image = cv.imread('input/100-dollars.tif' , 0)
height, width = image.shape
plane = np.zeros((height,width,8) , dtype=numpy.uint8)
mask = np.zeros((height , width) ,dtype=numpy.uint8)
cv.imshow("original" , image)
cv.imwrite("output/original.tif" , image)

bitplane(image, plane);

pic_reconstruct = reconstruct(plane[:,:,7] , plane[:,:,6] , plane[:,:,5])
cv.imshow("reconstruct", pic_reconstruct)
cv.imwrite("output/reconstruct.tif" , pic_reconstruct)

image2 = cv.imread('input/blurry_moon.tif')
image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
lap_image = laplacian_filter(image2)

imgplot = plt.imshow(lap_image, cmap="gray")
plt.show()
plt.imsave('output/laplacian.png',lap_image,cmap="gray")


cv.waitKey(0)
cv.destroyAllWindows()
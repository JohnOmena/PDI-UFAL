#%%
import cv2
from matplotlib import pyplot as plt
import numpy as np
#%%
img = cv2.imread('/home/johnomena/Pictures/cat.jpeg', cv2.IMREAD_GRAYSCALE)

img = cv2.GaussianBlur(img, (3,3), 0)

laplacian = cv2.Lapracian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)


#%%
cv2.normalize(laplacian, laplacian, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(sobelx, sobelx, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(sobely, sobely, 0, 255, cv2.NORM_MINMAX)

laplacian - np.asanyarray(laplacian, dtype='uint8')

plt.subplot(2, 2, 1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])


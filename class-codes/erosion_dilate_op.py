#%% Importação das bibliotecas
import cv2
from matplotlib import pyplot as plt
import numpy as np
#%% Abertura das imagens
img_gt = cv2.imread('/home/alunoic/Downloads/gt.tif', cv2.IMREAD_GRAYSCALE)
img_utk = cv2.imread('/home/alunoic/Downloads/utk.tif', cv2.IMREAD_GRAYSCALE)
img_j = cv2.imread('/home/alunoic/Downloads/j.png', cv2.IMREAD_GRAYSCALE)
img_finger = cv2.imread('/home/alunoic/Downloads/finger.tif', cv2.IMREAD_GRAYSCALE)


#%% Visualização das imagens

plt.subplot(121), plt.imshow(img_gt, cmap = 'gray')
plt.subplot(122), plt.imshow(img_utk, cmap = 'gray')

#%% Operações lógicas

img_or = cv2.bitwise_or(img_gt, img_utk)
img_and = cv2.bitwise_and(img_gt, img_utk)
img_not = cv2.bitwise_not(img_gt, img_utk)
img_xor = cv2.bitwise_xor(img_gt, img_utk)

plt.subplot(221), plt.imshow(img_or, cmap = 'gray')
plt.subplot(222), plt.imshow(img_and, cmap = 'gray')
plt.subplot(223), plt.imshow(img_not, cmap = 'gray')
plt.subplot(224), plt.imshow(img_xor, cmap = 'gray')

#%% Erosão e dilatação nas imagens
kernel = np.ones((3,3),np.uint8)
img_dilation = cv2.dilate(img_j,kernel,iterations = 1)
img_erosion = cv2.erode(img_dilation,kernel,iterations = 1)


plt.subplot(131), plt.imshow(img_j, cmap = 'gray')
plt.subplot(132), plt.imshow(img_dilation, cmap = 'gray')
plt.subplot(133), plt.imshow(img_erosion, cmap = 'gray')

#%% erosão, dilatação, abertura e fechamento

img_finger_dilation = cv2.dilate(img_finger,kernel,iterations = 1)
img_finger_erosion = cv2.erode(img_finger,kernel,iterations = 1)
img_finger_open = cv2.morphologyEx(img_finger, cv2.MORPH_OPEN, kernel)
img_finger_close = closing = cv2.morphologyEx(img_finger, cv2.MORPH_CLOSE, kernel)

plt.subplot(221), plt.imshow(img_finger_dilation, cmap = 'gray')
plt.subplot(222), plt.imshow(img_finger_erosion, cmap = 'gray')
plt.subplot(223), plt.imshow(img_finger_open, cmap = 'gray')
plt.subplot(224), plt.imshow(img_finger_close, cmap = 'gray')
 
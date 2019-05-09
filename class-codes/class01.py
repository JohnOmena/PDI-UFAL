#import os
#folder = '/home/alunoic/PDI-UFAL/screenshots/'
#os.path.join(folder, 'imagename.jpg')
#%% helpers
help(enumerate)

#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np

#%% Read a grayscale image
img = cv2.imread('/home/alunoic/PDI-UFAL/screenshots/cat.jpeg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
#%% Load a color image
img = cv2.imread('/home/alunoic/PDI-UFAL/screenshots/cat.jpeg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
plt.imshow(img)

#%%
rgb = cv2.split(img)

plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.title('R'), plt.imshow(rgb[0], cmap='gray')
plt.subplot(223), plt.title('R'), plt.imshow(rgb[1], cmap='gray')
plt.subplot(224), plt.title('R'), plt.imshow(rgb[2], cmap='gray')

plt.show()

#%% 
img = cv2.imread('/home/alunoic/PDI-UFAL/screenshots/cat.jpeg', cv2.IMREAD_GRAYSCALE)
plt.subplot(121), plt.title('Original'), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.title('Histogram'), plt.hist(img.ravel(), 256, [0, 256])
plt.show()

#%%
img = cv2.imread('/home/alunoic/PDI-UFAL/screenshots/cat.jpeg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()

#%% Creating images
img = np.ones((50, 50), dtype=np.int8)
plt.imshow(150*img, cmap='gray', vmin = 0, vmax = 255)
plt.show()

#%% Creating random images
img = np.ones((250, 250), dtype=np.uint8)
cv2.randu(img, 0, 255)
plt.subplot(121), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(122), plt.title('Histogram'), plt.hist(img.ravel(), 256, [0, 256])

plt.show()
#import os
folder = '/home/alunoic/Downloads'
#os.path.join(folder, 'imagename.jpg')
#%% helpers
help(enumerate)

#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

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
#%% Initialize a random color 
# O MATPLOTLIB usa o formato RGB enquanto o openCV usa oturo formato, deve-se usar a linha 65 para converter os canais. 
img = np.ones((15, 15, 3), dtype=np.uint8)
bgr = cv2.split(img)
cv2.randu(bgr[0], 0, 255)
cv2.randu(bgr[1], 0, 255)
cv2.randu(bgr[2], 0, 255)
img = cv2.merge(bgr)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.subplot(211), plt.imshow(img)
plt.subplot(212), plt.imshow(img2)
#%% Initialize a color image with random values, normaly distributed.
img = np.ones((250,250,3), np.uint8)
bgr = cv2.split(img)
cv2.randn(bgr[0], 127, 40)
cv2.randn(bgr[1], 127, 40)
cv2.randn(bgr[2], 127, 40)
img = cv2.merge(bgr)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(211), plt.imshow(img)
plt.subplot(212), plt.title('Histogram'), plt.hist(img.ravel(), 256, [0, 256])
#%%
img = np.ones((3, 3), dtype = np.float32)
cv2.randn(img, 0, 1)
print(img)
cv2.normalize(img, img, 255, 0, cv2.NORM_MINMAX)
print('Normalized = \n', img, '\n\n')
img = np.asarray(img, dtype = np.uint8)
print('Converted to uint8 = \n', img, '\n\n')
# Cast int ( ( (Z - Zmin) / (Zmax - Zmin) ) * 255 ) Como converter de uma faixa de valores para 255 
#%% Estudar translação em imagens a fundo para a prova.
# cv2.getRotationMatrix -> Olhar
# Matriz de rotação e translação para imagens. (Rotacionou e transladou)
img = cv2.imread(os.path.join(folder, 'lena.png'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]
quarter_height, quarter_width = height/4 , width/4
T = np.float32([
        [0,-1, quarter_width],
        [1,0, quarter_height],
        ])
img_translation = cv2.warpAffine(
        img, T, (width, height)
        )
plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(img_translation)
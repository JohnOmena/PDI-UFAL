import keras
keras.__version__
import matplotlib.pyplot as plt
import numpy as np
#%% tensowflow playground
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

model.summary()

#%%
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))
model.summary()

#%%
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()
idx = 10
plt.imshow(255 - train_images[idx], cmap = 'gray')
plt.xlabel('Class = ' + str(train_labels[idx]))
plt.show()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255


test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer = 'rmsprop', loss= 'categorical_crossentropy', metrics=['accuracy']) 
#%%
model.fit(train_images, train_labels, epochs = 5, batch_size = 64)
test_loss, test_acc = model.evaluate(test_images, test_labels)

#%%
pred_labels = model.predict_classes(test_images)
pred_scores = model.predict(test_images)
labels = np.argmax(test_labels, axis = 1)
idxs = ~(pred_labels == labels)
ind = np.where(idxs)[0]
for i in ind:
    img = test_images[i]
    plt.imshow(255 - img[:, :, 0], cmap = 'gray')
    plt.xlabel('Class = ' + str(labels[i]) + 'pred. class = ' + str(pred_labels[i]))
    plt.show()
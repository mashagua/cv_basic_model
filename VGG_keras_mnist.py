import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import cv2
import numpy as np
from keras.models import load_model
from keras.utils.vis_utils import plot_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def vgg16():
    # 初始化
    model = Sequential()
    "block_1"  # block1:
    # 64个卷积核
    model.add(ZeroPadding2D(padding=(1, 1), input_shape=( 28, 28,1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),data_format="channels_last"))
    "block_2"
    # 128个卷积核 conv_1,conv_2
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),data_format="channels_last"))
    """block_3"""
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last"))
    """block_4"""
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last"))
    '''
    "block_5"
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),data_format="channels_last"))
    '''
    ##
    model.add(Flatten())
    """FC_1"""
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.3))
    """FC_2"""
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.3))
    """分类"""
    model.add(Dense(10, activation='softmax'))
    model.save("my_model_vgg.h5")
    return model



model = vgg16()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
print(model.summary())
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
x_train=x_train.reshape([-1,28, 28,1])
x_test=x_test.reshape([-1,28,28,1])
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          verbose=1)
model.predict(x_test[0].reshape([-1,28,28,1]))
from PIL import Image
image=Image.open('K:/Edownload/ss.png').convert("L")
image=image.resize((28,28))
im2arr = np.array(image)
im2arr = im2arr.reshape(1,28,28,1)
    # Predicting the Test set results
y_pred = model.predict(im2arr)
print(y_pred)

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

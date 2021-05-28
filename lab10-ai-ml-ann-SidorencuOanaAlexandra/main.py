from multiprocessing.spawn import prepare
from sre_parse import CATEGORIES

import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import image
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.image as mpimg




def map_labels_to_oneHotLabels(data):
    maxim = max(data)+1
    l = []

    for i in range(len(data)):
        ll = [0]*maxim
        ll[data[i]] = 1
        l.append(ll)

    return l


def clasificare_flori():
    irisData = load_iris()
    all_data, labels = shuffle(irisData.data, irisData.target)
    #labels = map_labels_to_oneHotLabels(labels)
    labels = tf.keras.utils.to_categorical(labels,3)

    k = int(0.8*len(all_data))
    train_data = all_data[:k]
    train_output = labels[:k]

    valid_data = all_data[k:]
    valid_output = labels[k:]

    #normalizam datele
    scaler = StandardScaler()
    scaler.fit(train_data)
    valid_data = scaler.transform(valid_data)
    train_data = scaler.transform(train_data)

    model  = keras.Sequential([keras.layers.Dense(2,input_dim = 4,activation = tf.nn.relu),
                               keras.layers.Dense(3,activation=tf.nn.softmax)
                               ])

    model.compile(optimizer='adam', loss = keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

    model.summary()

    history = model.fit(train_data,train_output,validation_data=(valid_data,valid_output),batch_size=10,epochs=1000)
    print(history.history)
    print(history.history.keys())


    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1,len(loss)+1)
    plt.plot(epochs,loss,'y',label='Training loss')
    plt.plot(epochs,val_loss,'r',label = 'Validation loss')
    plt.plot(epochs,acc,color = 'black',label = 'Training Accuracy')
    plt.plot(epochs,val_acc,color = 'blue',label = 'Validation Accuracy')
    plt.title('Flori')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print(history.history['val_loss'])

def clasificare_numere():

    type_labels = ['0','1','2','3','4','5','6','7','8','9']

    #load data
    set = keras.datasets.mnist
    (trainImage,trainOutput),(validImage,validOutput) = set.load_data()
    lala = validImage[:10]

    #normalizare data
    trainImage,validImage = trainImage/255, validImage/255

    #transformare in one-hot encoding
    trainOutput = keras.utils.to_categorical(trainOutput,10)
    validOutput = keras.utils.to_categorical(validOutput,10)

    # trainImage = trainImage.reshape(trainImage.shape[0],28,28,1)
    # validImage = validImage.reshape(validImage.shape[0],28,28,1)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(123,activation='relu'),
        keras.layers.Dense(80,activation='relu'),
        keras.layers.Dense(50,activation='relu'),
        keras.layers.Dense(10,activation='softmax')
        # keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
        # keras.layers.Conv2D(64,(3,3),activation='relu'),
        # keras.layers.MaxPooling2D(pool_size=(2,2)),
        # keras.layers.Dropout(0.2),
        # keras.layers.Flatten(),
        # keras.layers.Dense(128,activation='relu'),
        # keras.layers.Dense(80,activation='relu'),
        # keras.layers.Dense(10,activation='softmax')



    ])
    # model = keras.Sequential()
    # model.add(Conv2D(filters=16,kernel_size=(5,5),activation="relu",input_shape=(28,28,1)))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    #
    # model.add(Flatten())
    # model.add(Dense(128,activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64,activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10,activation='softmax'))

    model.summary()

    model.compile(optimizer = keras.optimizers.SGD(),loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

    history = model.fit(trainImage,trainOutput,validation_data=(validImage,validOutput),epochs=40,batch_size=65)

    print(history.history.keys())

    loss = history.history['loss']
    loss_val = history.history['val_loss']
    acc = history.history['accuracy']
    acc_val = history.history['val_accuracy']
    epochs = range(1,len(loss)+1)
    plt.plot(epochs,loss,color='red',label = 'Training loss')
    plt.plot(epochs,loss_val,color='blue',label='Validation loss')
    plt.plot(epochs,acc,color='yellow',label='Training accuracy')
    plt.plot(epochs,acc_val,color='black',label='Validation accuracy')
    plt.title('Numere')
    plt.legend()
    plt.show()


    # SIZE=28
    # path = "D:\FACULTATE\An 2 Sem 2\Inteligenta Artificiala\Laburi\L10\Poze"
    # j=0
    # for root, dirs, files in os.walk(path):
    #     for i in files:
    #         img = keras.preprocessing.image.load_img(os.path.join(root,i),target_size=(SIZE,SIZE,1))
    #         img = img.convert(mode="1", dither=Image.NONE)
    #         #img = keras.image.rgb_to_grayscale(img)
    #         plt.imshow(img)
    #         img = keras.preprocessing.image.img_to_array(img)
    #         img=img/255.
    #         plt.show()
    #         j+=1
    #         print(j)
    #         img = np.expand_dims(img,axis=0)
    #         proba = model.predict(img)
    #         print(proba)
    #
    #
    #         max = 1000
    #         poz=0
    #         for l in range(len(proba[0])):
    #             if proba[0][l]<max:
    #                 max=proba[0][l]
    #                 poz=l
    #
    #         print("Trebuiesa sa fie" + str(i) + "a prezis ca este  "+str(poz))

def sepia(img):
    width, height = img.size

    pixels = img.load() # create the pixel map

    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))

            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            if tr > 255:
                tr = 255

            if tg > 255:
                tg = 255

            if tb > 255:
                tb = 255

            pixels[px, py] = (tr,tg,tb)

    return img

def data():
    path = 'D:\FACULTATE\An 2 Sem 2\Inteligenta Artificiala\Laburi\L10\lala\lala'
    data_images = []
    labels = []
    SIZE = 150
    l = 0
    for root, dirs, files in os.walk(path):
        for i in files:
            if l<300:
                img1 = image.load_img(os.path.join(root, i), target_size=(SIZE, SIZE, 3))
                img2 = image.load_img(os.path.join(root, i), target_size=(SIZE, SIZE, 3))

                img2 =sepia(img2)
                l+=1
                img1 = image.img_to_array(img1)
                img1 = img1 / 255.
                data_images.append(img1)
                labels.append(0)

                img2 = image.img_to_array(img2)
                img2 = img2 / 255.
                data_images.append(img2)
                labels.append(1)
            else:
                break

    return data_images,labels

def clasificare_imagini_sepia():

    path =  'D:\FACULTATE\An 2 Sem 2\Inteligenta Artificiala\Laburi\L10\Sepia'

    data_images, labels = data()
    # data_images = []
    SIZE = 150
    # for root, dirs, files in os.walk(path):
    #     for i in files:
    #         img = image.load_img(os.path.join(root,i),target_size=(SIZE,SIZE,3))
    #         img = image.img_to_array(img)
    #         img = img/255.
    #         data_images.append(img)

    print(len(data_images))
    k = int(0.8 * len(data_images))
    trainImage = data_images[:k]
    trainImage = np.array(trainImage)
    validImage = data_images[k:]
    validImage = np.array(validImage)

    labels = tf.keras.utils.to_categorical(labels,2)

    trainOutput = labels[:k]
    validOutput = labels[k:]


    # model = keras.Sequential()
    # model.add(Conv2D(filters=16,kernel_size=(5,5),activation="relu",input_shape=(SIZE,SIZE,3)))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    #
    # model.add(Flatten())
    # model.add(Dense(128,activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(64,activation='relu'))
    # model.add(Dense(2,activation='softmax'))
    #
    # model.summary()
    #
    #
    # model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    #
    # history = model.fit(trainImage,trainOutput,validation_data=(validImage,validOutput),epochs=30,batch_size=64)

    model = keras.Sequential([
        keras.layers.Conv2D(32,(5,5),activation='relu',input_shape=(SIZE,SIZE,3)),
        keras.layers.Conv2D(64,(5,5),activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(80,activation='relu'),
        keras.layers.Dense(2,activation='softmax')
    ])
    model.summary()

    model.compile(optimizer='adam',loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

    history = model.fit(trainImage,trainOutput,validation_data=(validImage,validOutput),epochs=15,batch_size=64)

    loss = history.history['loss']
    loss_val = history.history['val_loss']
    acc = history.history['accuracy']
    acc_val = history.history['val_accuracy']
    ep = range(1,len(loss)+1)
    plt.plot(ep,loss,color = 'red',label = "Training Loss")
    plt.plot(ep, loss_val, color='purple', label="Validation Loss")
    plt.plot(ep, acc, color='blue', label="Trainig accuracy")
    plt.plot(ep, acc_val, color='black', label="Validation accuracy")
    plt.legend()
    plt.title('Sepia')
    plt.show()



    path2 = 'D:\FACULTATE\An 2 Sem 2\Inteligenta Artificiala\Laburi\L10\Poze2'
    l=0
    for root, dirs, files in os.walk(path2):
        for i in files:
            img1 = image.load_img(os.path.join(root, i), target_size=(SIZE, SIZE, 3))
            img2 = img1
            img1 = image.img_to_array(img1)
            img1 = img1/255.
            img1 = np.expand_dims(img1,axis=0)
            proba = model.predict(img1)
            min = "normala"
            if proba[0][0]>proba[0][1]:
                print("Poza "+str(os.path.join(root, i))+ " este normala")
            else:
                print("Poza " + str(os.path.join(root, i)) + " este de tip sepia")
                min = "sepia"
            plt.imshow(img2)
            plt.xlabel(min)
            l+=1

            plt.show()






def main():

    clasificare_flori()
    clasificare_numere()
    clasificare_imagini_sepia()
    #data()


main()
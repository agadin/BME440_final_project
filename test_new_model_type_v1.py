import numpy as np
import seaborn as sns
import cv2
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.callbacks.reduce_lr_on_plateau import ReduceLROnPlateau
from keras.src.callbacks.tensorboard import TensorBoard
from keras.src.callbacks.model_checkpoint import ModelCheckpoint
from keras.src.utils.numerical_utils import to_categorical
from keras.src.layers.pooling.global_average_pooling2d import (
    GlobalAveragePooling2D,
)
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.core.dense import Dense
from keras.src.models.model import Model

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

sns.palplot(colors_dark)
sns.palplot(colors_green)
sns.palplot(colors_red)

labels = ['glioma','no_tumor','meningioma','pituitary']

X_train = []
y_train = []
image_size = 150

for i in labels:
    folderPath = os.path.join('/Users/colehanan/PycharmProjects/BME440_final_project/dataset', 'Training', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)

for i in labels:
    folderPath = os.path.join('/Users/colehanan/PycharmProjects/BME440_final_project/dataset', 'Testing', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train, y_train = shuffle(X_train,y_train, random_state=101)
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)

y_train_new = []

for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = to_categorical(y_train)
y_test_new = []

for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = to_categorical(y_test)

effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))

model = effnet.output
model = GlobalAveragePooling2D()(model)
model = Dropout(rate=0.5)(model)
model = Dense(4,activation='softmax')(model)
model = Model(inputs=effnet.input, outputs = model)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])

tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)

history = model.fit(X_train,y_train,validation_split=0.1, epochs =12, verbose=1, batch_size=32,
                   callbacks=[tensorboard,checkpoint,reduce_lr])

pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

print(classification_report(y_test_new,pred))

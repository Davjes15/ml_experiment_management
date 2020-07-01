"""
This sample script trains a basic CNN on the
Fashion-MNIST dataset. It takes black and white images of clothing
and labels them as "pants", "belt", etc.
"""

#Import comet libraries
from comet_ml import Experiment
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import random

# To run in beluga is necessary to 
#Comet can be used after loading the httpproxy module: module load httpproxy
#module load httpproxy

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="8cOIekzCAhWBHhOHEH1ADsa5S", project_name="general", workspace="davquispe")
# Add mutiple tags
experiment.add_tags(['test2','keras', 'cedar'])

# Set hyperparameters
dropout = 0.2
hidden_layer_size = 128
layer_1_size = 16
layer_2_size = 32
learn_rate = 0.001
decay = 1e-6
momentum = 0.9
epochs = 5
status=True

(X_train_orig, y_train_orig), (X_test, y_test) = fashion_mnist.load_data()

# Reducing the dataset size to 10,000 examples for faster train time
true = list(map(lambda x: True if random.random() < 0.167 else False, range(60000)))
ind = []
for i, x in enumerate(true):
    if x == True: ind.append(i)

X_train = X_train_orig[ind, :, :]
y_train = y_train_orig[ind]

img_width=28
img_height=28
labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

sgd = SGD(learning_rate=learn_rate, decay=decay, momentum=momentum, nesterov=status)

# build model
model = Sequential()
model.add(Conv2D(layer_2_size, (5, 5), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(layer_2_size, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#Add Keras callbacks
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=epochs, callbacks=my_callbacks)


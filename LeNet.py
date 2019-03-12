# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
import cPickle
import gzip
import numpy as np
seed = 7
np.random.seed(seed)
data = gzip.open(r'mnist.pkl.gz')
train_set, valid_set, test_set = cPickle.load(data)
train_x = train_set[0].reshape((-1,28,28,1))
train_y = to_categorical(train_set[1])
valid_x = valid_set[0].reshape((-1,28,28,1))
valid_y = to_categorical(valid_set[1])
test_x = test_set[0].reshape((-1,28,28,1))
test_y = to_categorical(test_set[1])

# creat model
model = Sequential()
model.add(Conv2D(32, (5, 5),strides=(1, 1), input_shape=(28, 28, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_x,train_y,validation_data=(valid_x, valid_y), atch_size=20, epochs=20, verbose=2)
print(model.evaluate(test_x, test_y, batch_size=20, verbose=2))
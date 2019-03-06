from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np
np.random.seed(1337)  # for reproducibility


#  载入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#  数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#  搭建神经网络
model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

#  优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall', 'fbeta_score'])

# 训练
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

# 没有验证集

# 测试
loss, accuracy, precision, recall, fbeta_score = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)
print('test precision:', precision)
print('test recall: ', recall)
print('test fbeta_score', fbeta_score)
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
model.add(Dense(units=10, input_shape=(n_input)))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(train_x, train_y, batch_size = batch_size, epochs = 1, validation_split = validation_split)
res = model.evaluate(test_x, test_y)
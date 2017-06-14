import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

trainingData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targetData = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(16, input_dim=2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["binary_accuracy"])

model.fit(trainingData, targetData, nb_epoch=5, verbose=2)

print(model.predict(trainingData).round())

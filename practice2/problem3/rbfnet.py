from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(-2,2,60)
out = x*x*x
list1 = []
for i in x:
	list1.append(np.asarray([math.e ** (-1 * (j - i)**2) for j in x]))

model = Sequential()
model.add(Dense(60, input_dim=1, activation='tanh'))
model.add(Dense(1))

sgd = SGD(lr=0.2)
model.compile(loss='mean_squared_error',
              optimizer=sgd)
for i in list1:
	model.fit(i, out, epochs=110)

res = np.asarray(model.predict(list1[-1]))

r = plt.figure(1)
plt.plot(x, res, '-o', label="True")
plt.plot(x, out, '-o', label="x^3")
plt.legend()
r.show()

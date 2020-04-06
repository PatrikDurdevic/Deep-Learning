import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import random


train_size = 10000
test_size = 100000
epochs = 30
max_number = 1e5

x_train = [random.randint(0, max_number) for i in range(train_size)]
y_train = [i + 1 for i in x_train]

x_train = tf.constant(x_train, shape=(train_size, 1))
y_train = tf.constant(y_train, shape=(train_size, 1))

x_test = [random.randint(0, max_number * 10) for i in range(test_size)]
y_test = [i + 1 for i in x_test]

model = tf.keras.Sequential([
	tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer='adam',
              loss=loss_fn)

model.fit(x_train, y_train, epochs=epochs)

model.evaluate(x_test,  y_test, verbose=2)

x = 1
print("Deep Learning model calculates x + 1")
while x > 0:
	x = int(input("Insert a number (the program terminates when x <= 0): "))
	y = model(tf.constant(x, shape=(1,1)))
	print(int(round(tf.keras.backend.get_value(y)[0][0])))

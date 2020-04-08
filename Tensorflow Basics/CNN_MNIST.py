import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
epochs = 5

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

model = keras.Sequential([
	keras.layers.Conv2D(filters=24, kernel_size=(3,3), activation=tf.nn.relu),
	keras.layers.MaxPool2D(pool_size=(2,2)),
	keras.layers.Conv2D(filters=36, kernel_size=(3,3), activation=tf.nn.relu),
	keras.layers.MaxPool2D(pool_size=(2,2)),

	keras.layers.Flatten(),
	keras.layers.Dense(128, activation=tf.nn.relu),

	keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()


def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array, true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	img = img.reshape((28, 28))
	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(predicted_label, 100*np.max(predictions_array), true_label), color=color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array, true_label[i]
	plt.grid(False)
	plt.xticks(range(10))
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

predictions = model.predict(test_images)

num_rows = 4
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, 2*i+1)
	plot_image(i, predictions[i], test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2*i+2)
	plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
















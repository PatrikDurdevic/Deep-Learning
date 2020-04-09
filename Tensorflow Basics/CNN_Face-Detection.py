import tensorflow as tf
from tensorflow import keras

import numpy as np
import h5py
import matplotlib.pyplot as plt

batch_size = 32
epochs = 2
filters = 12
train_part = 0.8

print("Downloading data...")
path_to_training_data = tf.keras.utils.get_file('train_face.h5', 'https://www.dropbox.com/s/l5iqduhe0gwxumq/train_face.h5?dl=1')
print("Downloaded data, parsing...")

f = h5py.File(path_to_training_data, 'r')
images = np.array(f.get('images'), dtype=np.float32)
print("Loaded images!")
labels = np.array(f.get('labels'), dtype=np.float32)
print("Loaded labels!")

print("Splitting data...")
split_point = int(images.shape[0] * train_part)
train_images, train_labels = images[:split_point], labels[:split_point]
test_images, test_labels = images[split_point:], labels[split_point:]

model = tf.keras.Sequential([
	keras.layers.Conv2D(filters=1*filters, kernel_size=5,  strides=2),
	keras.layers.BatchNormalization(),

	keras.layers.Conv2D(filters=2*filters, kernel_size=5,  strides=2),
	keras.layers.BatchNormalization(),

	keras.layers.Conv2D(filters=4*filters, kernel_size=3,  strides=2),
	keras.layers.BatchNormalization(),

	keras.layers.Conv2D(filters=6*filters, kernel_size=3,  strides=2),
	keras.layers.BatchNormalization(),

	keras.layers.Flatten(),
	keras.layers.Dense(512),
	keras.layers.Dense(1, activation=None),
])

def loss_fn(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

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
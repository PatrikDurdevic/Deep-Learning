import tensorflow as tf
from tensorflow import keras

import numpy as np
import h5py
import matplotlib.pyplot as plt

batch_size = 32
epochs = 2
filters = 12
latent_dim = 100
train_part = 0.8

print("Downloading data...")
# Positive training data: CelebA Dataset. A large-scale (over 200K images) of celebrity faces. (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
# Negative training data: ImageNet. Many images across many different categories. We'll take negative examples from a variety of non-human categories. (http://www.image-net.org/)
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

def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005):
	latent_loss = 0.5 * tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1.0 - logsigma, axis=1)
	reconstruction_loss = tf.reduce_mean(tf.abs(x-x_recon), axis=(1,2,3))

	vae_loss = kl_weight * latent_loss + reconstruction_loss

	return vae_loss

def sampling(z_mean, z_logsigma):
	batch, latent_dim = z_mean.shape
	epsilon = tf.random.normal(shape=(batch, latent_dim))

	z = z_mean + tf.math.exp(0.5 * z_logsigma) * epsilon
	return z

def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
	vae_loss = vae_loss_function(x, x_pred, mu, logsigma)  
	classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logit)

	face_indicator = tf.cast(tf.equal(y, 1), tf.float32)

	total_loss = tf.reduce_mean(
	  classification_loss + 
	  face_indicator * vae_loss
	)

	return total_loss, classification_loss
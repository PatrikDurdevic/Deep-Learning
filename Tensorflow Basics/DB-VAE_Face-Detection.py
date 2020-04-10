import tensorflow as tf
from tensorflow import keras

import numpy as np
import h5py
import matplotlib.pyplot as plt
import tqdm

batch_size = 32
epochs = 6
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

	total_loss = tf.reduce_mean(classification_loss + face_indicator * vae_loss)

	return total_loss, classification_loss

def make_face_encoder_network(output_size):
	encoder = tf.keras.Sequential([
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
		keras.layers.Dense(output_size, activation=None),
	])

	return encoder

def make_face_decoder_network():
	decoder = tf.keras.Sequential([
		keras.layers.Dense(units=4*4*6*filters, activation='relu'),
		keras.layers.Reshape(target_shape=(4, 4, 6*filters)),

		keras.layers.Conv2DTranspose(filters=4*filters, kernel_size=3,  strides=2, padding='same', activation='relu'),
		keras.layers.Conv2DTranspose(filters=2*filters, kernel_size=3,  strides=2, padding='same', activation='relu'),
		keras.layers.Conv2DTranspose(filters=1*filters, kernel_size=5,  strides=2, padding='same', activation='relu'),
		keras.layers.Conv2DTranspose(filters=3, kernel_size=5,  strides=2, padding='same', activation='relu'),
	])

	return decoder

class DB_VAE(tf.keras.Model):
	def __init__(self, latent_dim):
		super(DB_VAE, self).__init__()
		self.latent_dim = latent_dim

		num_encoder_dims = 2 * self.latent_dim + 1

		self.encoder = make_face_encoder_network(num_encoder_dims)
		self.decoder = make_face_decoder_network()

	def encode(self, x):
		encoder_output = self.encoder(x)

		y_logit = tf.expand_dims(encoder_output[:, 0], -1)

		z_mean = encoder_output[:, 1:self.latent_dim+1] 
		z_logsigma = encoder_output[:, self.latent_dim+1:]

		return y_logit, z_mean, z_logsigma

	def reparameterize(self, z_mean, z_logsigma):
		z = sampling(z_mean, z_logsigma)
		
		return z

	def decode(self, z):
		reconstruction = self.decoder(z)
		
		return reconstruction

	def call(self, x): 
		y_logit, z_mean, z_logsigma = self.encode(x)

		z = self.reparameterize(z_mean, z_logsigma)
		recon = self.decode(z)
		
		return y_logit, z_mean, z_logsigma, recon

	def predict(self, x):
		y_logit, z_mean, z_logsigma = self.encode(x)
		return y_logit

dbvae = DB_VAE(latent_dim)

def get_latent_mu(images, dbvae, batch_size=1024):
	N = images.shape[0]
	mu = np.zeros((N, latent_dim))
	for start_ind in range(0, N, batch_size):
		end_ind = min(start_ind+batch_size, N+1)
		batch = (images[start_ind:end_ind]).astype(np.float32)/255.
		_, batch_mu, _ = dbvae.encode(batch)
		mu[start_ind:end_ind] = batch_mu
	return mu

'''Function that recomputes the sampling probabilities for images within a batch
      based on how they distribute across the training data'''
def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=0.001): 
	print("Recomputing the sampling probabilities")

	mu = get_latent_mu(images, dbvae)
	# sampling probabilities for the images
	training_sample_p = np.zeros(mu.shape[0])

	# consider the distribution for each latent variable 
	for i in range(latent_dim):
		latent_distribution = mu[:,i]
		# generate a histogram of the latent distribution
		hist_density, bin_edges =  np.histogram(latent_distribution, density=True, bins=bins)

		# find which latent bin every data sample falls in 
		bin_edges[0] = -float('inf')
		bin_edges[-1] = float('inf')

		# call the digitize function to find which bins in the latent distribution 
		#    every data sample falls in to
		# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
		bin_idx = np.digitize(latent_distribution, bin_edges)

		# smooth the density function
		hist_smoothed_density = hist_density + smoothing_fac
		hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

		# invert the density function 
		p = 1.0/(hist_smoothed_density[bin_idx-1])

		# normalize all probabilities
		p = p / np.sum(p)

		# update sampling probabilities by considering whether the newly
		#     computed p is greater than the existing sampling probabilities.
		training_sample_p = np.maximum(p, training_sample_p)

	# final normalization
	training_sample_p /= np.sum(training_sample_p)

	return training_sample_p






dbvae = DB_VAE(100)
# TODO: SET LEARNING RATE!
optimizer = tf.keras.optimizers.Adam(0.002)

@tf.function
def debiasing_train_step(x, y):

	with tf.GradientTape() as tape:
		# Feed input x into dbvae. Note that this is using the DB_VAE call function!
		y_logit, z_mean, z_logsigma, x_recon = dbvae(x)
		# call the DB_VAE loss function to compute the loss
		loss, class_loss = debiasing_loss_function(x, x_recon, y, y_logit, z_mean, z_logsigma)

	# use the GradientTape.gradient method to compute the gradients.
	grads = tape.gradient(loss, dbvae.trainable_variables)
	# apply gradients to variables
	optimizer.apply_gradients(zip(grads, dbvae.trainable_variables))

	return loss

# get training faces from data loader
all_faces = train_images

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

# The training loop -- outer loop iterates over the number of epochs
for i in range(epochs):
	print("Starting epoch {}/{}".format(i+1, epochs))

	# Recompute data sampling proabilities
	p_faces = get_training_sample_probabilities(all_faces, dbvae)

	# get a batch of training data and compute the training step
	for j in tqdm(range(train_images.shape[0] // batch_size)):
		# load a batch of data
		#(x, y) = loader.get_batch(batch_size, p_pos=p_faces)
		(x, y) = train_images[p_faces * batch_size:(p_faces + 1) * batch_size]
		# loss optimization
		loss = debiasing_train_step(x, y)





































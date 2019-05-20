'''
Fully Connected auto encoder that imposes a gaussian prior
on latent representations.
'''
import numpy as np
import tensorflow as tf
import mnist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
tfd = tf.contrib.distributions

lat_dim = 2

print("Setting up Graph...")
data = tf.placeholder(tf.float32, [None,28,28])

flattened_in = tf.layers.flatten(data)
h1_e = tf.layers.dense(flattened_in,200,tf.nn.relu)
h2_e = tf.layers.dense(h1_e,100,tf.nn.relu)

means = tf.layers.dense(h2_e,lat_dim)
devs = tf.layers.dense(h2_e,lat_dim,tf.nn.softplus)

posterior = tfd.MultivariateNormalDiag(means,devs)
encoding = posterior.sample()

h1_d = tf.layers.dense(encoding,100,tf.nn.relu)
h2_d = tf.layers.dense(h1_d,200,tf.nn.relu)
flattened_out = tf.layers.dense(h2_d,784)

reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(flattened_in,flattened_out)))

prior = tfd.MultivariateNormalDiag(tf.zeros(lat_dim),tf.ones(lat_dim))
alpha = 0
KL_loss = alpha*tf.reduce_mean(tfd.kl_divergence(posterior,prior))

loss = reconstruction_loss+KL_loss

LEARNING_RATE = 1e-3
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

print("Loading Training Images")
training_images = mnist.train_images()
training_images = training_images.astype(float)/255.0
training_labels = mnist.train_labels()

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

EPOCHS = 10
MB_SIZE = 10
batches = len(training_images)//MB_SIZE
numV = 5

for i in range(EPOCHS):
	print("")
	print("Training Epoch", i)
	encodings = sess.run(encoding,feed_dict={data:training_images})
	plt.scatter(encodings[:,0],encodings[:,1],c=training_labels,cmap='tab10')
	plt.show()
	plt.close('all')
	rl,kl,l = sess.run((reconstruction_loss,KL_loss,loss),feed_dict={data:training_images})
	print("Loss:",l)
	print("reconstruction_loss:",rl)
	print("KL_loss:",kl)
	for b in range(batches):
		print(f"Batch: {b+1}",end='\r')
		if(b%3000==0):
			indices = np.random.choice(len(training_images),numV)
			images = training_images[indices]
			recs = np.reshape(sess.run(flattened_out,feed_dict={data:images}),[numV,28,28])
			for t in range(numV):
				plt.subplot(numV,2,t*2+1)
				plt.imshow(images[t],cmap='gray')
				plt.subplot(numV,2,t*2+2)
				plt.imshow(recs[t],cmap='gray')
			plt.show()
			plt.close('all')
		batch = training_images[b*MB_SIZE:(b+1)*MB_SIZE]
		sess.run(train_step,feed_dict={data:batch})
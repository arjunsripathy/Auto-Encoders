'''
Convolutional autoencoder that imposes a gaussion prior
on latent variables.
'''
import numpy as np
import tensorflow as tf
import mnist
import matplotlib.pyplot as plt
import sklearn.decomposition
from mpl_toolkits.mplot3d import Axes3D
tfd = tf.contrib.distributions

DISPLAY = True
USE_LAST = False

lat_dim = 8

print("Setting up Graph...")
data = tf.placeholder(tf.float32, [None,28,28])

c_1 = tf.layers.conv2d(inputs=tf.reshape(data,[-1,28,28,1]),filters=2,kernel_size=5)
#24x24x2

mp_1 = tf.layers.max_pooling2d(inputs=c_1,pool_size=2,strides=2)
#12x12x2

c_2 = tf.layers.conv2d(inputs=mp_1,filters=4,kernel_size=5)
#8x8x4

mp_2 = tf.layers.max_pooling2d(inputs=c_2,pool_size=2,strides=2)
#4x4x4

c_3 = tf.layers.conv2d(inputs=mp_2,filters=8,kernel_size=3)
#2x2x8

flat = tf.reshape(c_3,[-1,2*2*8])
#32

means = tf.layers.dense(flat,lat_dim)
devs = tf.layers.dense(flat,lat_dim,tf.nn.softplus)

posterior = tfd.MultivariateNormalDiag(means,devs)

encoding = posterior.sample()
#lat

u_flat = tf.layers.dense(encoding,2*2*8)
#32

t_1 = tf.layers.conv2d_transpose(inputs=tf.reshape(u_flat,[-1,2,2,8]),filters=4,kernel_size=3)
#2x2x8 -> 4x4x4

u_1 = tf.image.resize_images(t_1,size=(8,8),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#8x8x4

t_2 = tf.layers.conv2d_transpose(inputs=u_1,filters=2,kernel_size=5)
#12x12x2

u_2 = tf.image.resize_images(t_2,size=(24,24),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#24x24x2

t_3 = tf.layers.conv2d_transpose(inputs=u_2,filters=1,kernel_size=5)
#28x28x1

rec = tf.reshape(t_3,[-1,28,28])

reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(data,rec)))

prior = tfd.MultivariateNormalDiag(tf.zeros(lat_dim),tf.ones(lat_dim))
alpha = 0
KL_loss = alpha*tf.reduce_mean(tfd.kl_divergence(posterior,prior))

loss = reconstruction_loss+KL_loss

LEARNING_RATE = 5e-4
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

print("Loading Training Images")
training_images = mnist.train_images()
training_images = training_images.astype(float)/255.0
training_labels = mnist.train_labels()

sess = tf.InteractiveSession()

saver = tf.train.Saver()

if(USE_LAST):
	saver.restore(sess,'/tmp/model.ckpt')
else:
	sess.run(tf.global_variables_initializer())

EPOCHS = 10
MB_SIZE = 10
batches = len(training_images)//MB_SIZE
numV = 5

for i in range(EPOCHS):
	print("")
	print("Calculating Metrics")
	if(DISPLAY):
		encodings = sess.run(encoding,feed_dict={data:training_images})
		pca = sklearn.decomposition.PCA(n_components=2)
		pca.fit(encodings)
		p_comps = pca.transform(encodings)
		plt.scatter(p_comps[:,0],p_comps[:,1],c=training_labels,cmap='tab10')
		plt.show()
		plt.close('all')
	rl,kl,l = sess.run((reconstruction_loss,KL_loss,loss),feed_dict={data:training_images})
	print("Loss:",l)
	print("reconstruction_loss:",rl)
	print("KL_loss:",kl)
	print("Training Epoch", i+1)
	for b in range(batches):
		print(f"Batch: {b+1}",end='\r')
		if(DISPLAY and b%3000==0):
			indices = np.random.choice(len(training_images),numV)
			images = training_images[indices]
			recs = sess.run(rec,feed_dict={data:images})
			for t in range(numV):
				plt.subplot(numV,2,t*2+1)
				plt.imshow(images[t],cmap='gray')
				plt.subplot(numV,2,t*2+2)
				plt.imshow(recs[t],cmap='gray')
			plt.show()
			plt.close('all')
		batch = training_images[b*MB_SIZE:(b+1)*MB_SIZE]
		sess.run(train_step,feed_dict={data:batch})

saver.save(sess,'/tmp/model.ckpt')



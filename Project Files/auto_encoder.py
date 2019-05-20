'''
Convolutional autoencoder that imposes no constraints
on latent variables.
'''

import numpy as np
import tensorflow as tf
import mnist
import matplotlib.pyplot as plt
import sklearn.decomposition
tfd = tf.contrib.distributions

lat_dim = 10

DISPLAY = True
USE_LAST = False

PCA = True
BLOCK = False

MODE = 0

print("Setting up Graph...")
data = tf.placeholder(tf.float32, [None,28,28])

c_1 = tf.layers.conv2d(inputs=tf.reshape(data,[-1,28,28,1]),filters=8,kernel_size=5,activation=tf.nn.leaky_relu)
#24x24x8

mp_1 = tf.layers.max_pooling2d(inputs=c_1,pool_size=2,strides=2)
#12x12x8

c_2 = tf.layers.conv2d(inputs=mp_1,filters=16,kernel_size=5,activation=tf.nn.leaky_relu)
#8x8x16

mp_2 = tf.layers.max_pooling2d(inputs=c_2,pool_size=2,strides=2)
#4x4x16

enc = tf.layers.conv2d(inputs=mp_2,filters=lat_dim,kernel_size=4)
#1x1xlat_dim

flat = tf.reshape(enc,[-1,lat_dim])

t_0 = tf.layers.conv2d_transpose(inputs=enc,filters=16,kernel_size=4,activation=tf.nn.leaky_relu)
#4x4x16

u_1 = tf.layers.conv2d_transpose(inputs=t_0,filters=16,kernel_size=2,strides=2)
#8x8x16

t_1 = tf.layers.conv2d_transpose(inputs=u_1,filters=8,kernel_size=5,activation=tf.nn.leaky_relu)
#12x12x8

u_2 = tf.layers.conv2d_transpose(inputs=t_1,filters=8,kernel_size=2,strides=2)
#24x24x8

t_2 = tf.layers.conv2d_transpose(inputs=u_2,filters=1,kernel_size=5,activation=tf.nn.sigmoid)
#28x28x1

rec = tf.reshape(t_2,[-1,28,28])

boost = 1.1
loss = tf.reduce_mean(tf.square(tf.subtract(boost*data,rec)))
error = tf.reduce_mean(tf.square(tf.subtract(data,rec)))

LEARNING_RATE = 5e-5
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

print("Loading Images")
training_images = mnist.train_images()
training_images = training_images.astype(float)/255.0
training_labels = mnist.train_labels()

sess = tf.InteractiveSession()

saver = tf.train.Saver()

if(USE_LAST):
	saver.restore(sess,'/tmp/model.ckpt')
else:
	sess.run(tf.global_variables_initializer())

numV = 5

if(MODE==0):
	EPOCHS = 10
	MB_SIZE = 10
	batches = len(training_images)//MB_SIZE

	for e in range(EPOCHS):
		print("Training Epoch", e+1)
		for b in range(batches):
			print(f"Batch: {b+1}",end='\r')
			if(DISPLAY and b%1000==0):
				indices = np.random.choice(len(training_images),numV)
				images = training_images[indices]
				recs = sess.run(rec,feed_dict={data:images})
				for t in range(numV):
					plt.subplot(numV,2,t*2+1)
					plt.imshow(images[t],cmap='gray')
					plt.subplot(numV,2,t*2+2)
					plt.imshow(recs[t],cmap='gray')
				plt.show(block=False)
				plt.pause(1)
				plt.close('all')
			batch = training_images[b*MB_SIZE:(b+1)*MB_SIZE]
			sess.run(train_step,feed_dict={data:batch})
		print("")
		print("Calculating Metrics")
		if(DISPLAY):
			NUM_ENCODINGS = 4000
			encodings = sess.run(flat,feed_dict={data:training_images[:NUM_ENCODINGS]})
			if(PCA):
				n = 2
				pca = sklearn.decomposition.PCA(n_components=n)
				pca.fit(encodings)
				comps = pca.transform(encodings)
				plt.scatter(comps[:,0],comps[:,1],c=training_labels[:NUM_ENCODINGS],cmap='tab10')
			else:
				sums = [[[0,0] for _ in range(10)] for _ in range(lat_dim)]
				for i in range(lat_dim):
					for j in range(NUM_ENCODINGS):
						digit = training_labels[j]
						sums[i][digit][0]+=encodings[j][i]
						sums[i][digit][1]+=1
				cents = [[sums[i][j][0]/sums[i][j][1] for j in range(10)] for i in range(lat_dim)]

				x = sum(cents,[])
				y = sum([[i+1]*10 for i in range(lat_dim)],[])
				c = list(range(10))*lat_dim
				plt.scatter(x,y,c=c,cmap='tab10')
				plt.gca().set_ylim([0,lat_dim+1])
			if(BLOCK):
				plt.show()
			else:
				plt.show(block=False)
				plt.pause(1)
			plt.close('all')

		l = sess.run(error,feed_dict={data:training_images})
		print("Loss:",l)
		print("Saving Network")
		saver.save(sess,'/tmp/model.ckpt')

elif(MODE==1):
	while(True):
		indices = np.random.choice(len(training_images),numV)
		images = training_images[indices]
		img_mag = np.sum(np.square(images))
		for i in range(8):
			noise = np.random.normal(scale=0.1*i,size=[numV,28,28])
			noise_mag = np.sum(np.square(noise))
			noised_images = images+noise
			recs = sess.run(rec,feed_dict={data:noised_images})
			print(f"Noise in Images: {(noise_mag/(noise_mag+img_mag))*100}%")
			err = np.mean(np.abs(recs-images))
			print(f"Average Pixel Error: {(err**0.5)*100}%")
			print("")
			for t in range(numV):
				plt.subplot(numV,2,t*2+1)
				plt.imshow(noised_images[t],cmap='gray')
				plt.subplot(numV,2,t*2+2)
				plt.imshow(recs[t],cmap='gray')
			plt.show(block=False)
			plt.pause(3)
			plt.close('all')





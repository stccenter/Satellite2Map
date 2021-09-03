import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from matplotlib import pyplot
import os

strategy = tf.distribute.MirroredStrategy()
# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = tf.keras.initializers.RandomNormal(stddev=0.02)
	# source image input
	in_src_image = tf.keras.Input(shape=image_shape)
	# target image input
	in_target_image = tf.keras.Input(shape=image_shape)
	# concatenate images channel-wise
	merged = tf.keras.layers.Concatenate()([in_src_image, in_target_image])
	# C64
	d = tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
	# C128
	d = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = tf.keras.layers.BatchNormalization()(d)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
	# C256
	d = tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = tf.keras.layers.BatchNormalization()(d)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
	# C512
	d = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = tf.keras.layers.BatchNormalization()(d)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = tf.keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = tf.keras.layers.BatchNormalization()(d)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
	# patch output
	d = tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = tf.keras.layers.Activation('sigmoid')(d)
	# define model
	model = tf.keras.Model([in_src_image, in_target_image], patch_out)
	# compile model
	#opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
	#model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = tf.keras.initializers.RandomNormal(stddev=0.02)
	# add downsampling layer
	g = tf.keras.layers.Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = tf.keras.layers.BatchNormalization()(g, training=True)
	# leaky relu activation
	g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = tf.keras.initializers.RandomNormal(stddev=0.02)
	# add upsampling layer
	g = tf.keras.layers.Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = tf.keras.layers.BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = tf.keras.layers.Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = tf.keras.layers.Concatenate()([g, skip_in])
	# relu activation
	g = tf.keras.layers.Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = tf.keras.initializers.RandomNormal(stddev=0.02)
	# image input
	in_image = tf.keras.layers.Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = tf.keras.layers.Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = tf.keras.layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = tf.keras.layers.Activation('tanh')(g)
	# define model
	model = tf.keras.Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = tf.keras.Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = tf.keras.Model(in_src, [dis_out, gen_out])
	# compile model
	#opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
	#model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	X1Train, X2Train = X1[0:int(len(X1)*0.8)],X2[0:int(len(X2)*0.8)]
	X1Valid, X2Valid = X1[int(len(X1)*0.8):],X2[int(len(X2)*0.8):]
	# scale from [0,255] to [-1,1]
	X1Train = (X1Train - 127.5) / 127.5
	X2Train = (X2Train - 127.5) / 127.5
	X1Valid = (X1Valid - 127.5) / 127.5
	X2Valid = (X2Valid - 127.5) / 127.5
	return [X1Train, X2Train, X1Valid, X2Valid]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB, validA, validB  = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	#X1, X2 = trainA[tf.gather_nd(ix)], trainB[tf.convert_to_tensor(ix)]
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

def evaluate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB, validA, validB  = dataset
	# choose random instances
	#207 247 183 [ 207  247 183]
	#ix = randint(0, trainA.shape[0], n_samples)
	print("Fixed images")
	# retrieve selected images
	#X1, X2 = trainA[tf.gather_nd(ix)], trainB[tf.convert_to_tensor(ix)]
	X1, X2 = validA[0:int(len(validA))], validB[0:int(len(validA))]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	#X = g_model(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_epochs, n_batch, n_samples):
	print("summarize")
	# select a sample of input images
	[X_realA, X_realB], _ = evaluate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	origin_path = "multi_plot_origin_%d_%d" % (n_epochs, n_batch)
	prediction_path = "multi_plot_prediction_%d_%d" % (n_epochs, n_batch)
	label_path = "multi_plot_label_%d_%d" % (n_epochs, n_batch)
	access_rights = 0o777
	os.umask(0)
	os.mkdir(origin_path, access_rights)
	os.mkdir(prediction_path, access_rights)
	os.mkdir(label_path, access_rights)
	# plot real source images
	for i in range(n_samples):
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
		filename1 = '%d_multi_plot_origin_%d_%d.png' % (i, n_epochs, n_batch)
		pyplot.savefig(origin_path + "/" + filename1)
		pyplot.close()
	# plot generated target image
	for i in range(n_samples):
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
		filename2 = '%d_multi_plot_prediction_%d_%d.png' % (i, n_epochs, n_batch)
		pyplot.savefig(prediction_path + "/" + filename2)
		pyplot.close()
	# plot real target image
	for i in range(n_samples):
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
		filename3 = '%d_multi_plot_label_%d_%d.png' % (i, n_epochs, n_batch)
		pyplot.savefig(label_path + "/" + filename3)
		pyplot.close()
	# save the generator model
	filename4 = 'multi_model_%d_%d.h5' % (n_epochs, n_batch)
	g_model.save(filename4)
	print('Saved')

@tf.function
def train_step(d_model, g_model, X_realA, X_realB):    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        #generate a batch of fake samples
        X_fakeB = g_model(X_realA)        
        
        real_output = d_model([X_realA, X_realB], training = True)
        fake_output = d_model([X_realA, X_fakeB], training = True)
        
        #Calculate discriminator loss        
        d1_loss, d2_loss = discriminator_loss(real_output, fake_output)
        
        #Calculate generator loss using both labels and the target image
        g_loss = generator_loss(fake_output, X_fakeB, X_realB)
        
        d_loss = d1_loss + d2_loss       
    
    #Update the descriminator
    gradients_of_discriminator = disc_tape.gradient(d_loss, d_model.trainable_variables)
    
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
            
    gradients_of_generator = gen_tape.gradient(g_loss,g_model.trainable_variables) 
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))

    #tf.print('real_output:')
    #tf.print(real_output)
    #tf.print('fake_output:') 
    #tf.print(fake_output)
    #tf.print('g_loss:')
    #tf.print(g_loss)
    #tf.print('d1_loss:') 
    #tf.print(d1_loss)
    #tf.print('gradients_of_generator')
    #tf.print(gradients_of_generator)
    #tf.print('gradients_of_discriminator')
    #tf.print(gradients_of_discriminator)

    return d1_loss, d2_loss, g_loss

# prediction of 0 = fake, 1 = real
@tf.function
def discriminator_loss(real_output, fake_output):
    bce_real = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    real_loss = bce_real(tf.ones_like(real_output), real_output)
    
    bce_fake = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    fake_loss = bce_fake(tf.zeros_like(fake_output), fake_output)
        
    return 0.5*(real_loss)*(1./256.0), 0.5*(fake_loss)*(1./256.0)

@tf.function
def generator_loss(fake_output, X_fakeB, X_realB):
    bce_label = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    label_loss = bce_label(tf.ones_like(fake_output), fake_output)
    
    bce_img = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    img_loss = bce_img(X_realB, X_fakeB)
    

    return (label_loss)*(1./256.0) + img_loss

start = time.time()

n_epochs = 1000
batch_size = 8

with strategy.scope():

    # load image data
    dataset = np.array(load_real_samples('maps_256.npz'))
    print('Loaded', dataset[0].shape, dataset[1].shape, dataset[2].shape, dataset[3].shape)
    # define input shape based on the loaded dataset
    image_shape = dataset[0].shape[1:]
    # define the models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    # define the composite model
    #gan_model = define_gan(g_model, d_model, image_shape)

    generator_optimizer =  tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

    trainA, trainB, validA, validB = dataset
    datasetA = tf.data.Dataset.from_tensor_slices(trainA)
    datasetB = tf.data.Dataset.from_tensor_slices(trainB)
    datasetA = datasetA.cache()
    datasetB = datasetB.cache()
    datasetA = datasetA.batch(batch_size, drop_remainder=True)
    datasetB = datasetB.batch(batch_size, drop_remainder=True)
    datasetA = datasetA.prefetch(1)
    datasetB = datasetB.prefetch(1)

    datasetA = strategy.experimental_distribute_dataset(datasetA)
    datasetB = strategy.experimental_distribute_dataset(datasetB)

    # manually enumerate epochs
    for i in range(n_epochs):
        for batchA, batchB in zip(datasetA, datasetB):
            d1_loss, d2_loss, g_loss = strategy.run(train_step, args=(d_model, g_model, batchA, batchB))
            g_loss = (strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss, axis=None))/batch_size
            d1_loss = (strategy.reduce(tf.distribute.ReduceOp.SUM, d1_loss, axis=None))/batch_size
            d2_loss = (strategy.reduce(tf.distribute.ReduceOp.SUM, d2_loss, axis=None))/batch_size 
    
        print('>%d, d1_loss=%.3f, d2_loss = %.3f, gen_loss=%.3f' % (i, d1_loss, d2_loss, g_loss))
    summarize_performance(i, g_model, dataset, n_epochs, batch_size, int(len(validA)))

end = time.time()
print (str(end-start))

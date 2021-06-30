import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, MaxPooling2D, AveragePooling1D, Dropout, Concatenate, Conv2DTranspose, Reshape
from tensorflow.keras import Model
import numpy as np 
import keras
from keras import backend as K
class encoderOnePop(Model):
    def __init__(self, latent_dim):
        super(encoderOnePop, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')
        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(latent_dim*2) #two vectors of length latent_dim to represent means and standard deviations.

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.pool(x) # pool
        x = self.conv2(x)
        x = self.pool(x) # pool

        # note axis is 1 b/c first axis is batch
        # can try max or sum as the permutation-invariant function
        #x = tf.math.reduce_max(x, axis=1)
        x = tf.math.reduce_mean(x, axis=1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        return x 
    def input_size(self, x): 
        pass 

class decoderOnePop(Model):
    """Single population model - based on defiNETti software."""

    def __init__(self, input_shape):
        # the original input we aim to reconstruct
        super(decoderOnePop, self).__init__()
        self.conv1 = Conv2DTranspose(32, (1, 5), activation='relu', padding = 'same')
        self.conv2 = Conv2DTranspose(64, (1, 5), activation='relu', padding = 'same')
        self.conv3 = Conv2DTranspose(2, (1, 5), padding = 'same')
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(np.prod(input_shape), activation='relu')
        self.reshape = Reshape(target_shape = input_shape) 
    def call(self, x, training=None):
        """x is the latent space coordinates, should be of shape [latent_dim,]"""
        x = self.fc1(x)
        x = self.dropout(x, training = training)
        x = self.fc2(x) 
        x = self.dropout(x, training = training)
        x = self.reshape(x)
        x = self.conv1(x) 
        x = self.conv2(x)
        x = self.conv3(x)
        return x 

class encoderTwoPop(Model):
    def __init__(self, latent_dim, pop1, pop2):
        super(encoderTwoPop, self).__init__()

        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')
        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        self.merge = Concatenate()
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(latent_dim*2) # 2, activation='softmax') # two classes

        self.pop1 = pop1
        self.pop2 = pop2


    def call(self, x, training=None):
      x_pop1 = x[:, :self.pop1, :, :]
      x_pop2 = x[:, self.pop1:, :, :]

      # two conv layers for each part
      x_pop1 = self.conv1(x_pop1)
      x_pop2 = self.conv1(x_pop2)
      x_pop1 = self.pool(x_pop1) # pool
      x_pop2 = self.pool(x_pop2) # pool

      x_pop1 = self.conv2(x_pop1)
      x_pop2 = self.conv2(x_pop2)
      x_pop1 = self.pool(x_pop1) # pool
      x_pop2 = self.pool(x_pop2) # pool

      # 1 is the dimension of the individuals
      # can try max or sum as the permutation-invariant function
      #x_pop1_max = tf.math.reduce_max(x_pop1, axis=1)
      #x_pop2_max = tf.math.reduce_max(x_pop2, axis=1)
      x_pop1_sum = tf.math.reduce_mean(x_pop1, axis=1)
      x_pop2_sum = tf.math.reduce_mean(x_pop2, axis=1)

      # flatten all
      #x_pop1_max = self.flatten(x_pop1_max)
      #x_pop2_max = self.flatten(x_pop2_max)
      x_pop1_sum = self.flatten(x_pop1_sum)
      x_pop2_sum = self.flatten(x_pop2_sum)

      # concatenate
      m = self.merge([x_pop1_sum, x_pop2_sum]) # [x_pop1_max, x_pop2_max]
      m = self.fc1(m)
      m = self.dropout(m, training=training)
      m = self.fc2(m)
      m = self.dropout(m, training=training)
      return self.dense3(m)
    def input_size(self, x): 
        pass 

class decoderTwoPop(Model):
    """Single population model - based on defiNETti software."""

    def __init__(self, input_shape):
        # the original input we aim to reconstruct
        super(decoderTwoPop, self).__init__()
        self.conv1 = Conv2DTranspose(32, (1, 5), activation='relu', padding = 'same')
        self.conv2 = Conv2DTranspose(64, (1, 5), activation='relu', padding = 'same')
        self.conv3 = Conv2DTranspose(2, (1, 5), padding = 'same')
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(np.prod(input_shape), activation='relu')
        self.reshape = Reshape(target_shape = input_shape) 
    def call(self, x, training=None):
        """x is the latent space coordinates, should be of shape [latent_dim,]"""
        x = self.fc1(x)
        x = self.dropout(x, training = training)
        x = self.fc2(x) 
        x = self.dropout(x, training = training)
        x = self.reshape(x)
        x = self.conv1(x) 
        x = self.conv2(x)
        x = self.conv3(x)
        return x 
class CVAEOnePop(Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, input_shape):
    super(CVAEOnePop, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = encoderOnePop(latent_dim)
    self.decoder = decoderOnePop(input_shape) 

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

class CVAETwoPop(Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, input_shape, pop1, pop2):
    super(CVAETwoPop, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = encoderTwoPop(latent_dim, pop1, pop2)
    self.decoder = decoderTwoPop(input_shape) 

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  x = tf.divide(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z, apply_sigmoid = True)  
  reconstruction_loss = keras.losses.binary_crossentropy(x, x_logit, from_logits=True)
  reconstruction_loss = tf.reduce_mean(reconstruction_loss)
  kl_loss = 1 + logvar - K.square(mean) - K.exp(logvar)
  kl_loss = K.sum(kl_loss)
  kl_loss *= -0.5
  loss = tf.reduce_mean(reconstruction_loss + kl_loss)
  return loss

@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss 

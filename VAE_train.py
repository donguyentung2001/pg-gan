import math
import numpy as np 
import random
import sys
import tensorflow as tf
from scipy.optimize import basinhopping 
from VAE import decoderOnePop, encoderOnePop, CVAEOnePop, train_step  
import real_data_random
import util 

VAE_NUM_ITER = 70
VAE_BATCH_SIZE = 50
VAE_NUM_BATCH = 100
print("NUM_ITER", VAE_NUM_ITER)
print("BATCH_SIZE", VAE_BATCH_SIZE)
print("NUM_BATCH", VAE_NUM_BATCH)

# globals for data
NUM_SNPS = 36       # number of seg sites, should be divisible by 4
L = 50000           # heuristic to get enough SNPs for simulations
NUM_CLASSES = 2     # "real" vs "simulated"
NUM_CHANNELS = 2    # SNPs and distances
print("NUM_SNPS", NUM_SNPS)
print("L", L)
print("NUM_CLASSES", NUM_CLASSES)
print("NUM_CHANNELS", NUM_CHANNELS)

optimizer = tf.keras.optimizers.Adam(1e-4)

class VAE_train: 
    def __init__(self, iterator, model): 
        self.model = model 
        self.iterator = iterator 
    def training_sa(self, num_batches, batch_size): 
        for epoch in range(num_batches): 
            real_regions = self.iterator.real_batch(batch_size, True)
            loss = train_step(self.model, real_regions, optimizer)
            if (epoch+1) % 20 == 0:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1,loss))
        return loss 
    def run_training(self, num_iter = VAE_NUM_ITER): 
        for i in range(num_iter): 
            print("at iteration ", i+1)
            self.training_sa(VAE_NUM_BATCH, VAE_BATCH_SIZE) 



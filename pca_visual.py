import math
import numpy as np 
import random
import sys
from scipy.optimize import basinhopping 
import real_data_random
import util 
from sklearn.decomposition import PCA
import pandas as pd 


# globals for data
NUM_SNPS = 36       # number of seg sites, should be divisible by 4
L = 50000           # heuristic to get enough SNPs for simulations
NUM_CLASSES = 2     # "real" vs "simulated"
NUM_CHANNELS = 2    # SNPs and distances
print("NUM_SNPS", NUM_SNPS)
print("L", L)
print("NUM_CLASSES", NUM_CLASSES)
print("NUM_CHANNELS", NUM_CHANNELS)

def main(): 
    opts = util.parse_args()
    print(opts)
    iterator = process_opts(opts)
    x = iterator.real_batch(10, True)
    x = tf.divide(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
    #normalize 
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    print(principalDf)


def process_opts(opts): 
    if opts.data_h5 != None:
        # most typical case for real data
        iterator = real_data_random.RealDataRandomIterator(NUM_SNPS, \
            opts.data_h5, opts.bed)
        num_samples = iterator.num_samples # TODO use num_samples below
    return iterator 


if __name__ == '__main__': 
    main()
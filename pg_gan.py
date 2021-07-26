"""
Application entry point for PG-GAN.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date 2/4/21
"""

# python imports
import datetime
import math
import numpy as np
import random
import sys
import tensorflow as tf
from scipy.optimize import basinhopping
from VAE_train import VAE_train 
# our imports
import discriminators
import real_data_random
import simulation
import util
import matplotlib.pyplot as plt
from real_data_random import Region
from VAE import *
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, MaxPooling2D, AveragePooling1D, Dropout, Concatenate, Conv2DTranspose, Reshape
from tensorflow.keras import Model
# globals for simulated annealing
NUM_ITER = 1
BATCH_SIZE = 50
NUM_BATCH = 100
print("NUM_ITER", NUM_ITER)
print("BATCH_SIZE", BATCH_SIZE)
print("NUM_BATCH", NUM_BATCH)

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
    """Parse args and run simulated annealing"""

    opts = util.parse_args()
    print(opts)

    # set up seeds
    if opts.seed != None:
        np.random.seed(opts.seed)
        random.seed(opts.seed)
        tf.random.set_seed(opts.seed)

    generator, discriminator, iterator, VAE_model, parameters = process_opts(opts)

    # grid search
    if opts.grid:
        print("Grid search not supported right now")
        sys.exit()
        #posterior, loss_lst = grid_search(discriminator, samples, simulator, \
        #    iterator, parameters, opts.seed)
    # simulated annealing
    else:
        posterior, loss_lst = simulated_annealing(generator, discriminator,\
            iterator, VAE_model, parameters, opts.seed, toy=opts.toy)

    print(posterior)
    print(loss_lst)

def process_opts(opts):

    # parameter defaults
    all_params = util.ParamSet()
    parameters = util.parse_params(opts.params, all_params) # desired params
    param_names = [p.name for p in parameters]

    # if real data provided
    real = False
    if opts.data_h5 != None:
        # most typical case for real data
        iterator = real_data_random.RealDataRandomIterator(NUM_SNPS, \
            opts.data_h5, opts.bed)
        num_samples = iterator.num_samples # TODO use num_samples below
        real = True

    filter = False # for filtering singletons

    # parse model and simulator
    if opts.model == 'const':
        sample_sizes = [198]
        discriminator = discriminators.OnePopModel()
        simulator = simulation.simulate_const
        sample_region = iterator.real_batch(1, True)
        input_shape= sample_region.shape[1:]

        VAE_model = VAE_train(iterator, CVAEOnePop(2, input_shape))
        #print("FILTERING SINGLETONS")
        #filter = True

    # exp growth
    elif opts.model == 'exp':
        sample_sizes = [198] 
        discriminator = discriminators.OnePopModel()
        simulator = simulation.simulate_exp
        sample_region = iterator.real_batch(1, True)
        input_shape= sample_region.shape[1:]
        print("INPUT SHAPE IS ")
        print(input_shape)
        VAE_model = VAE_train(iterator, CVAEOnePop(2, input_shape))
        #print("FILTERING SINGLETONS")
        #filter = True

    # isolation-with-migration model (2 populations)
    elif opts.model == 'im':
        sample_sizes = [98,98]
        discriminator = discriminators.TwoPopModel(sample_sizes[0], \
            sample_sizes[1])
        simulator = simulation.simulate_im
        sample_region = iterator.real_batch(1, True)
        input_shape= sample_region.shape[1:]
        VAE_model = VAE_train(iterator, CVAETwoPop(2, input_shape, sample_sizes[0], sample_sizes[1]))

    # out-of-Africa model (2 populations)
    elif opts.model == 'ooa2':
        sample_sizes = [98,98]
        discriminator = discriminators.TwoPopModel(sample_sizes[0], \
            sample_sizes[1])
        simulator = simulation.simulate_ooa2
        sample_region = iterator.real_batch(1, True)
        input_shape= sample_region.shape[1:]
        VAE_model = VAE_train(iterator, CVAETwoPop(2, input_shape, sample_sizes[0], sample_sizes[1]))

    # CEU/CHB (2 populations)
    elif opts.model == 'post_ooa':
        sample_sizes = [98,98]
        discriminator = discriminators.TwoPopModel(sample_sizes[0], \
            sample_sizes[1])
        simulator = simulation.simulate_postOOA
        sample_region = iterator.real_batch(1, True)
        input_shape= sample_region.shape[1:]
        VAE_model = VAE_train(iterator, CVAETwoPop(2, input_shape, sample_sizes[0], sample_sizes[1]))

    # out-of-Africa model (3 populations)
    elif opts.model == 'ooa3':
        sample_sizes = [66,66,66]
        #per_pop = int(num_samples/3) # assume equal
        discriminator = discriminators.ThreePopModel(sample_sizes[0], \
            sample_sizes[1], sample_sizes[2])
        simulator = simulation.simulate_ooa3

    # no other options
    else:
        sys.exit(opts.model + " is not recognized")

    # generator
    generator = simulation.Generator(simulator, param_names, sample_sizes,\
        NUM_SNPS, L, opts.seed, mirror_real=real, reco_folder=opts.reco_folder,\
        filter=filter)

    # "real data" is simulated wiwh fixed params
    if opts.data_h5 == None:
        iterator = simulation.Generator(simulator, param_names, sample_sizes, \
            NUM_SNPS, L, opts.seed, filter=filter) # don't need reco_folder

    return generator, discriminator, iterator, VAE_model, parameters

################################################################################
# SIMULATED ANNEALING
################################################################################

def simulated_annealing(generator, discriminator, iterator, VAE_model, parameters, seed, \
    toy=False):
    """Main function that drives GAN updates"""

    # main object for pg-gan
    pg_gan = PG_GAN(generator, discriminator, iterator, VAE_model, parameters, seed)

    # find starting point through pre-training (update generator in method)
    if not toy:
        s_current = pg_gan.disc_pretraining(800, BATCH_SIZE)
    else:
        s_current = [param.start() for param in pg_gan.parameters]
        pg_gan.generator.update_params(s_current)

    loss_curr = pg_gan.generator_loss(s_current)
    print("params, loss", s_current, loss_curr)

    posterior = [s_current]
    loss_lst = [loss_curr]
    real_acc_lst = []
    fake_acc_lst = []

    # simulated-annealing iterations
    num_iter = NUM_ITER
    # for toy example
    if toy:
        num_iter = 2

    # main pg-gan loop
    for i in range(num_iter):
        print("\nITER", i)
        print("time", datetime.datetime.now().time())
        T = temperature(i, num_iter) # reduce width of proposal over time

        # propose 10 updates per param and pick the best one
        s_best = None
        loss_best = float('inf')
        for k in range(len(parameters)): # trying all params!
            #k = random.choice(range(len(parameters))) # random param
            for j in range(10): # trying 10

                # can update all the parameters at once, or choose one at a time
                #s_proposal = [parameters[k].proposal(s_current[k], T) for k in\
                #    range(len(parameters))]
                s_proposal = [v for v in s_current] # copy
                s_proposal[k] = parameters[k].proposal(s_current[k], T)
                loss_proposal = pg_gan.generator_loss(s_proposal)

                print(j, "proposal", s_proposal, loss_proposal)
                if loss_proposal < loss_best: # minimizing loss
                    loss_best = loss_proposal
                    s_best = s_proposal

        # decide whether to accept or not (reduce accepting bad state later on)
        if loss_best <= loss_curr: # unsure about this equal here
            p_accept = 1
        else:
            p_accept = (loss_curr / loss_best) * T
        rand = np.random.rand()
        accept = rand < p_accept

        # if accept, retrain
        if accept:
            print("ACCEPTED")
            s_current = s_best
            generator.update_params(s_current)
            # train only if accept
            real_acc, fake_acc = pg_gan.train_sa(NUM_BATCH, BATCH_SIZE)
            loss_curr = loss_best

        # don't retrain
        else:
            print("NOT ACCEPTED")

        print("T, p_accept, rand, s_current, loss_curr", end=" ")
        print(T, p_accept, rand, s_current, loss_curr)
        posterior.append(s_current)
        loss_lst.append(loss_curr)
    
    #visualize_filters(discriminator, "after_training")
    #feature_map_visualization(discriminator, iterator, "feature_map_aftertraining", "CEU", "CHB") 
    for i in range(4): 
        print(influential_nodes(self.discriminator, i)) 
    return posterior, loss_lst

def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob."""
    return 1 - i/num_iter # start at 1, end at 0

# not used right now
"""
def grid_search(model_type, samples, demo_file, simulator, iterator, \
        parameters, is_range, seed):

    # can only do one param right now
    assert len(parameters) == 1
    param = parameters[0]

    all_values = []
    all_likelihood = []
    for fake_value in np.linspace(param.min, param.max, num=30):
        fake_params = [fake_value]
        model = TrainingModel(model_type, samples, demo_file, simulator, \
            iterator, parameters, is_range, seed)

        # train more for grid search
        model.train(fake_params, NUM_BATCH*10, BATCH_SIZE)
        test_acc, conf_mat = model.test(fake_params, NUM_TEST)
        like_curr = likelihood(test_acc)
        print("params, test_acc, likelihood", fake_value, test_acc, like_curr)

        all_values.append(fake_value)
        all_likelihood.append(like_curr)

    return all_values, all_likelihood
"""

################################################################################
# TRAINING
################################################################################

class PG_GAN:

    def __init__(self, generator, discriminator, iterator, VAE_model, parameters, seed):
        """Setup the model and training framework"""
        print("parameters", type(parameters), parameters)


        # set up generator and discriminator
        self.generator = generator
        self.discriminator = discriminator
        self.iterator = iterator # for training data (real or simulated)
        self.parameters = parameters
        self.VAE_model = VAE_model

        # this checks and prints the model (1 is for the batch size)
        self.discriminator.build_graph((1, iterator.num_samples, NUM_SNPS, \
            NUM_CHANNELS))
        self.discriminator.summary()

        self.cross_entropy =tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.disc_optimizer = tf.keras.optimizers.Adam()

    def disc_pretraining(self, num_batches, batch_size):
        """Pre-train so discriminator has a chance to learn before generator"""
        '''s_best = []
        max_acc = 0
        k = 0

        # try with several random sets at first
        while max_acc < 0.9 and k < 10:
            s_trial = [param.start() for param in self.parameters]
            print("trial", k+1, s_trial)
            self.generator.update_params(s_trial)
            real_acc, fake_acc = self.train_sa(num_batches, batch_size)
            avg_acc = (real_acc + fake_acc)/2
            if avg_acc > max_acc:
                max_acc = avg_acc
                s_best = s_trial
            k += 1

        # now start!
        self.generator.update_params(s_best)
        return s_best''' 
        print("pretraining starts")
        self.VAE_model.run_training()
        trained_encoder = self.VAE_model.model.encoder
        print("The number of layers in encoder is ", len(trained_encoder.layers))
        print("The number of layers in discriminator is ", len(self.discriminator.layers))
        for i in range(len(trained_encoder.layers)):
            print("changing discriminator's weights in layer ", i)
            trained_encoder_weights = trained_encoder.layers[i].get_weights()
            self.discriminator.layers[i].set_weights(trained_encoder_weights)
            if i == 8: 
                print("pretraining weights are: ")
                print(trained_encoder_weights)
        print("finish pretraining with VAE. The discriminator layers should now be updated. \n Now we find the best parameters for the discriminators. ")
        for i in range(4): 
            print(influential_nodes(self.discriminator, i)) 
        #visualize_filters(trained_encoder, "pretraining")
        #feature_map_visualization(trained_encoder, self.VAE_model.iterator, "feature_map_pretraining", "CEU", "CHB") 
        #try either 10 times or when acc is 90% for the discriminator with simulated data
        max_acc = 0 
        k = 0
        while max_acc < 0.9 and k < 10:
            s_trial = [param.start() for param in self.parameters]
            self.generator.update_params(s_trial)
            print("trial", k+1, s_trial)
            real_acc, fake_acc = self.test_accuracy(BATCH_SIZE*NUM_BATCH)
            avg_acc = (real_acc + fake_acc)/2
            print("Accuracy now is", avg_acc) 
            if avg_acc > max_acc:
                max_acc = avg_acc
                s_best = s_trial
                print("New max accuracy is", max_acc)
            k += 1
        self.generator.update_params(s_best)
        return s_best
    def train_sa(self, num_batches, batch_size):
        """Train using fake_values for the simulated data"""

        for epoch in range(num_batches):

            real_regions = self.iterator.real_batch(batch_size, True)
            real_acc, fake_acc, disc_loss = self.train_step(real_regions)

            if (epoch+1) % 100 == 0:
                template = 'Epoch {}, Loss: {}, Real Acc: {}, Fake Acc: {}'
                print(template.format(epoch + 1,
                                disc_loss,
                                real_acc/BATCH_SIZE * 100,
                                fake_acc/BATCH_SIZE * 100))

        return real_acc/BATCH_SIZE, fake_acc/BATCH_SIZE

    def generator_loss(self, proposed_params):
        """ Generator loss """
        generated_regions = self.generator.simulate_batch(BATCH_SIZE, \
            params=proposed_params)
        # not training when we use the discriminator here
        fake_output = self.discriminator(generated_regions, training=False)
        loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        return loss.numpy()

    def discriminator_loss(self, real_output, fake_output):
        """ Discriminator loss """
        # accuracy
        real_acc = np.sum(real_output >= 0) # positive logit => pred 1
        fake_acc = np.sum(fake_output <  0) # negative logit => pred 0

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        # add on entropy regularization (small penalty)
        real_entropy = self.cross_entropy(real_output, real_output)
        fake_entropy = self.cross_entropy(fake_output, fake_output)
        entropy = tf.math.scalar_mul(0.001/2, tf.math.add(real_entropy, \
            fake_entropy)) # can I just use +,*? TODO
        total_loss -= entropy # maximize entropy

        return total_loss, real_acc, fake_acc

    def train_step(self, real_regions):
        """One mini-batch for the discriminator"""

        with tf.GradientTape() as disc_tape:
            # use current params
            generated_regions = self.generator.simulate_batch(BATCH_SIZE)

            real_output = self.discriminator(real_regions, training=True)
            fake_output = self.discriminator(generated_regions, training=True)

            disc_loss, real_acc, fake_acc = self.discriminator_loss( \
                real_output, fake_output)

        # gradient descent
        gradients_of_discriminator = disc_tape.gradient(disc_loss, \
            self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, \
            self.discriminator.trainable_variables))

        return real_acc, fake_acc, disc_loss

    def test_accuracy(self, batch_size): 
        with tf.GradientTape() as disc_tape:
            # use current params
            real_regions = self.iterator.real_batch(batch_size, True)
            generated_regions = self.generator.simulate_batch(BATCH_SIZE)

            real_output = self.discriminator(real_regions, training=True)
            fake_output = self.discriminator(generated_regions, training=True)

            disc_loss, real_acc, fake_acc = self.discriminator_loss( \
                real_output, fake_output)
        return real_acc/batch_size, fake_acc/batch_size

def visualize_filters(model, name): 
    for layer in model.layers: 
        if "conv" in layer.name: 
            # get filter weights
            filters, biases = layer.get_weights()
            print(layer.name, filters.shape)
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            filters = tf.squeeze(filters) #get rid of the first dimension since it is 1. 
            filters = tf.transpose(filters) #put the number of filters first, the number of channels second. and the filter weights last.  
            fig, ax = plt.subplots(int(filters.shape[0]/8), 8) 
            for i,ax_row in enumerate(ax):
                for j,axes in enumerate(ax_row):
                    print("filter being printed is number ", i*8+j)
                    current_filter = filters[i*8+j][0] 
                    current_filter = tf.expand_dims(current_filter, axis=0)
                    axes.set_yticks([])
                    axes.set_xticks([])
                    axes.imshow(current_filter, cmap='gray')
            plot_name = layer.name + name
            plt.savefig(plot_name)

def feature_map_visualization(model, iterator, plot_name, pop1, pop2): 
    temporary_model = tf.keras.Sequential() 
    temporary_model.add(Conv2D(32, (1, 5), activation='relu', input_shape = (98, 36, 2)))
    trained_encoder_weights = model.layers[0].get_weights()
    temporary_model.layers[0].set_weights(trained_encoder_weights)
    real_regions = iterator.real_batch(1, True)
    print("real region shape is ", real_regions.shape)
    print("the one we get has shape ", real_regions[:,:98,:,:].shape)
    feature_maps1 = temporary_model.predict(real_regions[:,:98,:,:])
    bool_tensor = tf.math.equal(real_regions[:,:98,:,:],real_regions[:,98:,:,:])
    print(tf.reduce_sum(tf.cast(bool_tensor, tf.float32)))
    print(feature_maps1.shape)
    feature_maps1 = tf.squeeze(feature_maps1)
    current_map = feature_maps1[:,:, 0] 
    print('current map is ', 0)
    #plt.set_yticks([])
    #plt.set_xticks([])
    plt.imshow(current_map, cmap='gray')
    plt.savefig(plot_name + pop1)
    feature_maps2 = temporary_model.predict(real_regions[:,98:,:,:])
    print(feature_maps2.shape)
    feature_maps2 = tf.squeeze(feature_maps2)
    current_map = feature_maps1[:,:, 0] 
    print('current map is ', 0)
    #plt.set_yticks([])
    #plt.set_xticks([])
    plt.imshow(current_map, cmap='gray')
    plt.savefig(plot_name + pop2)
    temporary_model2 = tf.keras.Sequential() 
    temporary_model2.add(Conv2D(32, (1, 5), activation='relu', input_shape = (98, 36, 2)))
    temporary_model2.add(Conv2D(64, (1, 5), activation='relu', input_shape = (98, 32, 32)))
    for i in range(2):
        trained_encoder_weights = model.layers[i].get_weights()
        temporary_model2.layers[i].set_weights(trained_encoder_weights)
    feature_maps3 = temporary_model2.predict(real_regions[:,:98,:,:])
    print(feature_maps3.shape)
    feature_maps3 = tf.squeeze(feature_maps3)
    current_map = feature_maps3[:,:, 0] 
    print('current map is ', 0)
    #plt.set_yticks([])
    #plt.set_xticks([])
    plt.imshow(current_map, cmap='gray')
    plt.savefig(plot_name + pop1 + "2")
    feature_maps4 = temporary_model2.predict(real_regions[:,98:,:,:])
    print(feature_maps4.shape)
    feature_maps4 = tf.squeeze(feature_maps4)
    current_map = feature_maps4[:,:, 0] 
    print('current map is ', 0)
    #plt.set_yticks([])
    #plt.set_xticks([])
    plt.imshow(current_map, cmap='gray')
    plt.savefig(plot_name + pop2 + "2")

def influential_nodes(model, k):
    output = { 
        8: (k, model.layers[8].get_weights()[1][k])  
    }
    previous_node_index = k 
    for i in range(7,4, -1): 
        print("Finding most influential nodes in layer ", i)
        current_layer = model.layers[i].get_weights()[1]
        print(current_layer)
        max_weight_node = max(current_layer, key= lambda x: abs(x[previous_node_index]))
        max_weight_index = current_layer.index(max_weight_node)
        output[i] = (max_weight_index, max_weight_node)
        print("appending node's index with weights", (max_weight_index, max_weight_node))
    return output 
if __name__ == "__main__":
    main()

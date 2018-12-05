import tensorflow as tf
import pandas as pd
import os
import json
from tensorboard.backend.event_processing import event_accumulator as ea
# import subprocess
# import time
import numpy as np
from collections import OrderedDict
from tabulate import tabulate
from pandas import ExcelWriter
import random
import math

import pprint as pprint

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D

import copy
import argparse

# os.chdir('/Users/blakecuningham/Dropbox/MScDataScience/Thesis/EDEN')

import sys
sys.path.append('/Users/blakecuningham/Dropbox/MScDataScience/Thesis/EDEN')

from evolution_helpers import *


print(new_generation_main('list_pop_20181028-N-1.json', 2))


#
# # Path variables
#
# # os.getcwd()
# os.chdir('/Users/blakecuningham/Dropbox/MScDataScience/Thesis/rl_openai_uni_starter/universe-starter-agent')
#
# user_path = '/Users/blakecuningham/'
#
# EDEN_dir = '../../EDEN/test_models/fixed_params/'
#
# # Get argument commands
# parser = argparse.ArgumentParser(description="Get the model file")
# parser.add_argument('-m', '--model-list', default=None,
#                     help="List of models for generation")
#
# args = parser.parse_args()
# model_list = args.model_list
#
#
#
#
# # model_list = 'list_pop_20181028-N-1.json'
# model_list_dir = EDEN_dir + model_list
#
# ## Fixed parameters
# # layers_conv = ['conv_2d', 'conv_2d', 'max_pool_2d', 'conv_2d', 'conv_2d', 'max_pool_2d', 'fully_conn']
# # layers_conn = ['fully_conn', 'fully_conn']
#
# # Functions
# #
# # def initialise_one_hidden_layer(layer_type, can_use_dropout):
# #     # Initialise an empty gene (dictionary)
# #     gene = {}
# #
# #     # Randomly generate a layer
# #     layer = generate_one_layer_type(layer_type, can_use_dropout)
# #
# #     # Set the name property of the gene
# #     gene["name"] = layer
# #
# #     # ---------------------------------------------------------------------------------------
# #     # If layer is full connected, then we need the number of units and an activation function.
# #     # ---------------------------------------------------------------------------------------
# #     if (layer == "fully_conn"):
# #
# #         # Set the units property for the fully connected layer
# #         gene["units"] = generate_num_units()
# #
# #         # Set the activation function for the fully connected layer
# #         gene["activation"] = generate_activation()
# #
# #     # ---------------------------------------------------------------------------------------
# #     # If the layer is dropout, then we need the keep probability.
# #     # ---------------------------------------------------------------------------------------
# #     elif (layer == "dropout"):
# #
# #         # Set the keep probability
# #         gene['keep_prob'] = generate_keep_prob()
# #
# #     # ---------------------------------------------------------------------------------------
# #     # If the layer is maxpool, then we need the kernel size.
# #     # ---------------------------------------------------------------------------------------
# #     elif (layer == "max_pool_2d"):
# #
# #         # Set the kernel size
# #         gene['kernel_size'] = generate_max_pooling_size()
# #
# #     # ---------------------------------------------------------------------------------------
# #     # If the layer is maxpool, then we need the kernel size.
# #     # ---------------------------------------------------------------------------------------
# #     elif (layer == "max_pool_1d"):
# #
# #         # Set the kernel size
# #         gene['kernel_size'] = generate_max_pooling_size_1D()
# #
# #     # ---------------------------------------------------------------------------------------
# #     # If the layer is convolution 2D, then we need the activation function, number of filters,
# #     # and the filter size.
# #     # ---------------------------------------------------------------------------------------
# #     elif (layer == "conv_2d"):
# #
# #         # Set the activation function
# #         gene["activation"] = generate_activation_conv()
# #
# #         # Set the number of filters
# #         gene["nb_filter"] = generate_number_filters()
# #
# #         # Set the filter size
# #         gene["filter_size"] = generate_filter_size()
# #         # ---------------------------------------------------------------------------------------
# #     # If the layer is convolution 1D, then we need the activation function, number of filters,
# #     # and the filter size.
# #     # ---------------------------------------------------------------------------------------
# #     elif (layer == "conv_1d"):
# #
# #         # Set the activation function
# #         gene["activation"] = generate_activation_conv()
# #
# #         # Set the number of filters
# #         gene["nb_filter"] = generate_number_filters()
# #
# #         # Set the filter size
# #         gene["filter_size"] = generate_filter_size_1D()
# #
# #
# #         # Return the newly created gene
# #     return gene
# #
# # def generate_one_layer_type(layer_type, can_use_dropout):
# #     """ Generate one layer.
# #
# #     TFAW Controller must decide, for a given problem, whether to use convolutional nets, fully connected, dropout or
# #     max pooling layers. The types of layers depends on the problem. For CNNs, we can make up the layers using those
# #     available in layers_conv. For non-CNNs, and also the part of CNNs which no longer use convolutional types of
# #     layers then we use those layers available in layers_conn.
# #
# #     Args:
# #         layer_type: either 'cnn' or 'non-cnn'.
# #         can_use_dropout: True implies the random selection can include dropout, False implies that the
# #         random selection cannot include dropout.
# #     Returns:
# #         A layer type (this is returned in the form of the name of the layer).
# #     """
# #
# #     # Convolutional types of layers
# #     # -----------------------------
# #     if (layer_type == 'cnn'):
# #
# #         if (can_use_dropout == True):
# #             # Get a random index for a layer from whatever is available (i.e. whatever is in 'layers_conv')
# #             index = random.randrange(len(layers_conv))
# #         else:
# #             # Get a random index for a layer from whatever is available (i.e. whatever is in 'layers_conv')
# #             # Here we add 1 so that the dropout in index position 0 is never selected.
# #             index = random.randrange(1, len(layers_conv))
# #
# #         # Return the layer type
# #         return layers_conv[index]
# #
# #     # Non-convolutional types of layers
# #     # -----------------------------
# #     else:
# #
# #         if (can_use_dropout == True):
# #             # Get a random index for a layer from whatever is available (i.e. whatever is in 'layelayers_connrs_conv')
# #             index = random.randrange(len(layers_conn))
# #         else:
# #
# #             # Get a random index for a layer from whatever is available (i.e. whatever is in 'layers_conn')
# #             # Here we add 1 so that the dropout in index position 0 is never selected.
# #             index = random.randrange(1, len(layers_conn))
# #
# #             # Return the layer type
# #         return layers_conn[index]
# #
# # def generate_activation_conv():
# #     index = random.randrange(3)
# #     if (index ==0):
# #         return "elu" # linear
# #     elif (index == 1):
# #         return "leaky_relu"
# #     elif (index == 1):
# #         return "prelu"
# #     else:
# #         return "relu"
# #
# # def generate_number_filters():
# #     # return random.randrange(10, 100)
# #     return random.randrange(24, 48)
# #
# # def generate_filter_size():
# #     # TODO change back to (1, 6)
# #     # size = random.randrange(2, 3)
# #     size = 3
# #     return size
# #
# # def generate_max_pooling_size():
# #     # size = random.randrange(1, 6)
# #     # TODO: temp for initial exploration
# #     size = 2
# #     return size
# #
# # def generate_num_units():
# #     # return random.randrange(10, 100)
# #     return random.randrange(24, 48)
# #
# # def generate_activation():
# #     index = random.randrange(4)
# #     if (index ==0):
# #         return "elu" # linear
# #     elif (index == 1):
# #         return "sigmoid"
# #     elif (index == 2):
# #         return "softmax"
# #     else:
# #         return "relu"
# #
# # def compute_parameters(architecture):
# #
# #     try:
# #
# #         number_of_layers = len(architecture)
# #
# #         # Create a sequential model using Keras
# #         model = Sequential()
# #
# #         # Extract the first layer
# #         first_layer = architecture[0]
# #
# #         if (architecture[0].get('name') == 'conv_2d'):
# #             # In this case we use input_shape=...
# #             model.add(Conv2D(input_shape=construct_input_shape_keras('cnn'),
# #                              filters=first_layer.get('nb_filter'), kernel_size=first_layer.get('filter_size'), padding='same', strides=(2, 2)))
# #
# #         elif (architecture[0].get('name') == 'max_pool_2d'):
# #             # print('b')
# #             # In this case we use input_shape=...
# #             model.add(MaxPooling2D(input_shape=construct_input_shape_keras('cnn'), pool_size=first_layer.get('kernel_size'), padding='same'))
# #
# #         elif (architecture[0].get('name') == 'fully_conn'):
# #             # print('c')
# #             # In this case we use input_dim=...
# #             model.add(Dense(units=first_layer.get('units'), input_dim=construct_input_shape_keras('non_cnn')))
# #
# #             # Skip the first layer since that has already been taken into consideration
# #         for i in range(1, number_of_layers):
# #
# #             # Get the corresponding layer in terms of Keras
# #             get_layer = get_layer_from_keras(architecture[i])
# #
# #             # If it is not 'None' then we can add the layer, i.e. it contains trainable parameters
# #             if (get_layer != None):
# #                 if(architecture[i].get('name') == 'fully_conn' and architecture[i-1].get('name') != 'fully_conn'):
# #                     model.add(Flatten())
# #                     model.add(get_layer)
# #                 else:
# #                     model.add(get_layer)
# #
# #         params = model.count_params()
# #         final_layer_params = int(np.prod(model.layers[-1].output_shape[1:]))
# #         del model
# #
# #     except Exception as e:
# #         print("Error caught (compute_parameters): ", e)
# #         return -1
# #
# #     # Return the number of trainable parameters
# #     return (params, final_layer_params)
# #
# # def construct_input_shape():
# #     '''Constructs the input shape for the DNN.
# #     '''
# #
# #     # Note: Manually matched shape to OpenAI rescale
# #
# #     # Initialise the shape to be empty
# #     shape = [None, 42, 42, 1]
# #
# #     # # The first element in the array must be "None" to indicate the batches
# #     # shape.append(None)
# #     #
# #     # # Iterate over the len of X_train shape (minus 1 because the first index in the array denotes
# #     # # the number of instances).
# #     # for i in range(0, len(X_train.shape) - 1):
# #     #     # Append the corresponding shape value from X_train.
# #     #     # We have to skip the first element in the shape so that's why
# #     #     # we start from index 1.
# #     #     shape.append(X_train.shape[i + 1])
# #
# #     return shape
# #
# # def construct_input_shape_keras(type):
# #     if (type == 'cnn'):
# #         # return (construct_input_shape()[1], construct_input_shape()[2], construct_input_shape()[3])
# #         return (42, 42, 1)
# #     else:
# #         return np.prod(construct_input_shape()[1:])
# #
# # def print_chromosome_details(chromosome):
# #     print("Optimiser: ",chromosome[0])
# #     print("Learning_rate: ", chromosome[1])
# #     print("#_parameters: ",chromosome[2])
# #     print("Validation_err: ",chromosome[3])
# #     print('Test_err: ', chromosome[5])
# #     print("Fitness: ", chromosome[4])
# #     print("Layers:")
# #     pprint.pprint(chromosome[6])
# #
# # def get_layer_from_keras(layer):
# #     ''' Get the layer in the format required by Keras.
# #         Read in a layer from the chromosome, and convert it into the correct format so that
# #         it can be added into Keras' sequential model. This helps to compute the number of
# #         trainable parameters in terms of the DNN parameters.
# #     '''
# #
# #     # Get the layer name
# #     layer_name = layer.get('name')
# #
# #     if (layer_name == 'fully_conn'):
# #         # Return a fully connected layer andthe only information needed to compute the weights is the
# #         # number of units in the layer.
# #         #TODO: Add flatten
# #         return Dense(units=layer.get('units'))
# #
# #     elif (layer_name == 'conv_2d'):
# #         # Return a convolutional 2d layer along with the number of filters and the kernel size
# #         return Conv2D(filters=layer.get('nb_filter'), kernel_size=layer.get('filter_size'), padding='same', strides=(2, 2))
# #
# #
# #     elif (layer_name == 'max_pool_2d'):
# #         # There are no DNN parameters which result from this operation
# #         return MaxPooling2D(pool_size=layer.get('kernel_size'), padding = 'same')
# #
# #     # Default return None
# #     else:
# #         return None
# #
# # def generate_learn_rate():
# #     """ Generate learning rate.
# #
# #     Edited to capture a wider range of rates around the original 1e-4
# #
# #     Generate a random float between 0.0 and 0.01. We add +1e-5 so that the function can never
# #     return a value of 0.
# #     """
# #
# #     # choice = random.randrange(4)
# #     # if (choice == 0):
# #     #     return round((random.random() / 1e2) + 1e-6, 6)
# #     # elif (choice == 1):
# #     #     return round((random.random() / 1e3 + 1e-6), 6)
# #     # elif (choice == 2):
# #     #     return round((random.random() / 1e4 + 1e-6), 6)
# #     # elif (choice == 3):
# #     #     return round((random.random() / 1e5 + 1e-6), 6)
# #
# #     return random.lognormvariate(0, 1) / 1e4
#
#
# # test_model_name = 'chromo20181021-LR-75e-05-P-32128-run-5-N-0'
#
# # def load_population_json(pop_list_path):
# #
# #     with open(model_list_dir) as json_data:
# #         model_list_data = json.load(json_data)
# #
# #     return model_list_data
# #
# # def model_output_to_pdframe(model_tested):
# #
# #     model_name_for_directory = model_tested[0].replace('.p','').replace('.','')
# #
# #     ## set log directory
# #
# #     # log_directory_main = user_path + 'Documents/temp_RL_output/'
# #     log_directory_main = '/Volumes/blakecjdl/RL_output/'
# #     log_directory = log_directory_main + model_name_for_directory
# #
# #     train0_directory = log_directory.replace('~/','') + '/train_0'
# #
# #     for file in os.listdir(train0_directory):
# #         log_file = file
# #
# #     eventsfile_path = train0_directory + '/' + log_file
# #
# #     acc = ea.EventAccumulator(eventsfile_path)
# #     acc.Reload()
# #
# #     ep_reward = [(s.step, s.value) for s in acc.Scalars('global/episode_reward')]
# #     ep_length = [(s.step, s.value) for s in acc.Scalars('global/episode_length')]
# #     ep_time = [(s.step, s.value) for s in acc.Scalars('global/episode_time')]
# #     reward_df = pd.DataFrame(ep_reward, columns = ('step', 'ep_reward'))
# #     length_df = pd.DataFrame(ep_length, columns = ('step', 'ep_length'))
# #     time_df = pd.DataFrame(ep_time, columns = ('step', 'ep_time'))
# #     reward_df['EMA'] = reward_df['ep_reward'].ewm(alpha = 0.01).mean()
# #     merged_df_1 = pd.merge(reward_df, length_df, on = ['step'])
# #     merged_df_2 = pd.merge(merged_df_1, time_df, on = ['step'])
# #
# #     return merged_df_2
# #
# # def model_name(model_tested):
# #
# #     model_name_for_directory = model_tested[0].replace('.p','').replace('.','')
# #
# #     return model_name_for_directory
# #
# # def model_results_summary_for_tournament(model_tested):
# #
# #     name_of_model = model_name(model_tested)
# #
# #     try:
# #         test_model_results_df =  model_output_to_pdframe(model_tested)
# #     except:
# #         test_model_results_df = pd.DataFrame({'step': [np.nan],
# #                                                 'ep_reward': [-21],
# #                                                 'EMA': [-21],
# #                                                 'ep_length': [np.nan],
# #                                                 'ep_time': [np.nan]})
# #
# #     ## The best episode in the experiment
# #     # resultstable_bestepisode = test_model_results_df['ep_reward'].max()
# #
# #     ## The number of times the result of the best episode was achieved
# #     # resultstable_bestepisodecount = np.sum(test_model_results_df['ep_reward'] == test_model_results_df['ep_reward'].max())
# #
# #     ## The best running score (excluding first 50 episodes to remove initial random bias)
# #     resultstable_bestEMA = test_model_results_df['EMA'][50:].max()
# #
# #     ## Unbiased variance and standard deviation of series
# #     # test_model_results_df['ep_reward'].var()
# #     # resultstable_overallSTD = test_model_results_df['ep_reward'].std()
# #
# #     ## Noise around EMA
# #
# #     noise_std = (test_model_results_df['ep_reward'] - test_model_results_df['EMA']).std()
# #
# #     ## General speed metric (MA distance / step)
# #
# #     # np.max((test_model_results_df['EMA'][100:] + 21) / test_model_results_df['step'][100:])
# #
# #     ## Tournament score
# #     try:
# #         tournament_score = (1 - ((resultstable_bestEMA + 21) / 42) +
# #                             0.5*(1- 1/(1 + noise_std)) +
# #                             0.5*(1 - 1/(1 + test_model_results_df['ep_time'].mean())))
# #     except:
# #         tournament_score = np.nan
# #
# #     resultstable_vector = OrderedDict()
# #     # resultstable_vector['Model'] = [str(name_of_model)]
# #     resultstable_vector['Best EMA'] = [round(resultstable_bestEMA,2)]
# #     resultstable_vector['Noise around EMA'] = [round(noise_std, 2)]
# #     resultstable_vector['Avg. episode time'] = test_model_results_df['ep_time'].mean()
# #     resultstable_vector['Tournament score'] = [round(tournament_score, 2)]
# #
# #     return pd.DataFrame(resultstable_vector)
# #
#
# population_of_models = load_population_json(model_list_dir)
#
# # print(population_of_models[0])
#
# # TODO: Add fail reason
# # TODO: Add mastery level
# # model_results_summary(population_of_models[3])
#
# overall_results_frame = pd.DataFrame()
#
# for model in population_of_models:
#     overall_results_frame = overall_results_frame.append(model_results_summary_for_tournament(model), ignore_index=True)
#
# # print(overall_results_frame)
# # overall_results_frame_sorted = overall_results_frame.sort_values(['Best EMA'], ascending=[False])
#
# # overall_results_frame.loc[[1]]['Tournament score'].values[0]
#
# # random.sample(range(len(overall_results_frame)), tournament_size)
#
# # def tournament_select_round_winner(results_frame, tournament_size):
# #     contenders = random.sample(range(len(results_frame)), tournament_size)
# #     current_best_score = math.inf
# #     for i in contenders:
# #         fitness = overall_results_frame.loc[[i]]['Tournament score'].values[0]
# #         if fitness < current_best_score:
# #             current_best_score = fitness
# #             current_best_chromo = i
# #
# #     return current_best_chromo
#
# # tournament_select_winner(overall_results_frame, 5)
#
# # population_of_models[tournament_select_winner(overall_results_frame, 5)]
#
# tournament_size = 3
# new_generation_size = 10
#
# # def select_tournament_winners(tournament_size, new_generation_size):
# #
# #     winner_chromosomes = []
# #
# #     # Build new generation of n parents?
# #
# #     for i in range(new_generation_size):
# #         winner_chromosomes.append(
# #             population_of_models[
# #             tournament_select_round_winner(overall_results_frame, tournament_size)])
# #
# #     return winner_chromosomes
#
# # del winner_chromosomes
#
# winner_chromosomes = select_tournament_winners(tournament_size, new_generation_size, population_of_models, overall_results_frame)
#
#
# # def generate_chromosome_change_operator(chromosome):
# #
# #     parent_chromosome = chromosome
# #
# #     num_chromosome_layers = len(parent_chromosome[1][6])
# #
# #     ## Change a layer of Chromosome
# #
# #     chromosome_layer_to_change = random.choice(range(num_chromosome_layers))
# #
# #     # generate layer
# #
# #     if ((parent_chromosome[1][6][chromosome_layer_to_change - 1].get('name') == 'fully_conn')
# #         and (chromosome_layer_to_change - 1 >= 0)):
# #             layer_type = 'not_cnn'
# #     else:
# #         layer_type = 'cnn'
# #
# #     valid = False
# #     loop_limit = 0
# #
# #     while not valid and loop_limit <= 1000:
# #
# #         loop_limit += 1
# #
# #         child_chromosome = copy.deepcopy(parent_chromosome)
# #
# #         # The first layer cannot use dropout
# #         if (chromosome_layer_to_change == 0):
# #             can_use_dropout = False
# #
# #         # Subsequent layers can use dropout
# #         else:
# #             can_use_dropout = True
# #
# #         #TODO: Add if statement here to allow full conn in last layer
# #         if (chromosome_layer_to_change >=4) and random.random() >= 0.5:
# #             new_layer = initialise_one_hidden_layer('not_cnn', can_use_dropout)
# #         else:
# #             new_layer = initialise_one_hidden_layer(layer_type, can_use_dropout)
# #
# #         # Add new layer into chromosome
# #
# #         child_chromosome[1][6][chromosome_layer_to_change] = new_layer
# #
# #         # Compute and store the number of NN parameters
# #         number_of_parameters_result = compute_parameters(child_chromosome[1][6])
# #
# #         # Check to see if the model is invalid
# #         if (number_of_parameters_result == -1):
# #
# #             # If it returns an exception and a value of -1, then the function should
# #             # try generate another chromosome
# #             valid = False
# #         else:
# #
# #             # If there is no exception then assign the number of parameters
# #             child_chromosome[1][2] = number_of_parameters_result
# #             valid = True
# #
# #     # Return the randomly generated chromosome
# #     return child_chromosome
# #
# # def generate_chromosome_delete_operator(chromosome):
# #
# #     parent_chromosome = chromosome
# #
# #     num_chromosome_layers = len(parent_chromosome[1][6])
# #
# #     ## Change a layer of Chromosome
# #
# #     chromosome_layer_to_delete = random.choice(range(num_chromosome_layers))
# #
# #     child_chromosome = copy.deepcopy(parent_chromosome)
# #
# #     del child_chromosome[1][6][chromosome_layer_to_delete]
# #
# #     # Compute and store the number of NN parameters
# #     number_of_parameters_result = compute_parameters(child_chromosome[1][6])
# #
# #     child_chromosome[1][2] = number_of_parameters_result
# #
# #     return child_chromosome
# #
# # def generate_chromosome_add_operator(chromosome):
# #
# #     parent_chromosome = chromosome
# #
# #     num_chromosome_layers = len(parent_chromosome[1][6])
# #
# #     ## Change a layer of Chromosome
# #
# #     chromosome_layer_to_add = random.choice(range(num_chromosome_layers + 1))
# #
# #     # generate layer
# #
# #     if ((parent_chromosome[1][6][chromosome_layer_to_change - 1].get('name') == 'fully_conn')
# #         and (chromosome_layer_to_change - 1 >= 0)):
# #             layer_type = 'not_cnn'
# #     else:
# #         layer_type = 'cnn'
# #
# #     valid = False
# #     loop_limit = 0
# #
# #     while not valid and loop_limit <= 1000:
# #
# #         loop_limit += 1
# #
# #         child_chromosome = copy.deepcopy(parent_chromosome)
# #
# #         # The first layer cannot use dropout
# #         if (chromosome_layer_to_change == 0):
# #             can_use_dropout = False
# #
# #         # Subsequent layers can use dropout
# #         else:
# #             can_use_dropout = True
# #
# #         #TODO: Add if statement here to allow full conn in last layer
# #         if (chromosome_layer_to_change >=4) and random.random() >= 0.5:
# #             new_layer = initialise_one_hidden_layer('not_cnn', can_use_dropout)
# #         else:
# #             new_layer = initialise_one_hidden_layer(layer_type, can_use_dropout)
# #
# #         # Add new layer into chromosome
# #
# #         # child_chromosome[1][6][chromosome_layer_to_change] = new_layer
# #
# #         child_chromosome[1][6].insert(chromosome_layer_to_add, new_layer)
# #
# #         # Compute and store the number of NN parameters
# #         number_of_parameters_result = compute_parameters(child_chromosome[1][6])
# #
# #         # Check to see if the model is invalid
# #         if (number_of_parameters_result == -1):
# #
# #             # If it returns an exception and a value of -1, then the function should
# #             # try generate another chromosome
# #             valid = False
# #         else:
# #
# #             # If there is no exception then assign the number of parameters
# #             child_chromosome[1][2] = number_of_parameters_result
# #             valid = True
# #
# #     # Return the randomly generated chromosome
# #     return child_chromosome
# #
# # def mutate_learn_rate(learn_rate):
# #
# #     # return np.round(np.average([generate_learn_rate(), learn_rate, learn_rate]),5)
# #     # return round(np.max([learn_rate + (random.lognormvariate(0, 1) * learn_rate) - (np.exp(1) * learn_rate), 1e-6]), 6)
# #     return round(np.max([learn_rate + (random.normalvariate(0, 1) * learn_rate), 1e-6]), 6)
# #
# # def mutate_chromosome(chromosome, run_number):
# #
# #     parent_chromosome = chromosome
# #     num_chromosome_layers = len(parent_chromosome[1][6])
# #
# #     # Decide type of layer mutation
# #
# #     if num_chromosome_layers == 1:
# #         mutation_type = random.choice(['add', 'change'])
# #     elif num_chromosome_layers >= 6:
# #         mutation_type = random.choice(['delete', 'change'])
# #     else:
# #         mutation_type = random.choice(['add', 'delete', 'change'])
# #
# #     # Apply layer mutation
# #
# #     if mutation_type == 'add':
# #         child_chromosome = generate_chromosome_add_operator(parent_chromosome)
# #     elif mutation_type == 'delete':
# #         child_chromosome = generate_chromosome_delete_operator(parent_chromosome)
# #     elif mutation_type == 'change':
# #         child_chromosome = generate_chromosome_change_operator(parent_chromosome)
# #
# #     # Possibly mutate learning rate
# #
# #     if random.random() > 0.5:
# #         child_chromosome[1][1] = mutate_learn_rate(child_chromosome[1][1])
# #
# #
# #     # Add generation number (and in genesis) and somehow include run number here
# #     # Do I want to inherit date and run number?
# #     # Why am I not generating file name from chromosome attributes rather?
# #
# #     chromodate = parent_chromosome[0].split('-')[0]
# #     overallrunattempt = parent_chromosome[0].split('N-')[1].split('-')[0].split('.')[0]
# #
# #     chromo_file_name = (chromodate +
# #                         '-LR-' + str(child_chromosome[1][1]) +
# #                         '-P-' + str(child_chromosome[1][2][0]) + '-O-' + str(child_chromosome[1][2][1]) +
# #                         '-run-' + str(run_number) + '-N-' + str(overallrunattempt) + '.p')
# #
# #     child_chromosome[0] = chromo_file_name
# #
# #     return child_chromosome
#
# # mutate_chromosome(winner_chromosomes[0], 24)
#
# last_run_from_previous_gen = population_of_models[-1][0].split('run-')[1].split('-')[0]
#
# # def create_full_new_generation(winning_parents, last_run_number):
# #
# #     run_number = int(last_run_number)
# #
# #     new_generation_chromosomes_pt1 = []
# #
# #     for chromo in winning_parents:
# #         new_generation_chromosomes_pt1.append(mutate_chromosome(chromo, run_number))
# #         run_number += 1
# #
# #     new_generation_chromosomes_pt2 = []
# #
# #     for chromo in new_generation_chromosomes_pt1:
# #         new_generation_chromosomes_pt2.append(mutate_chromosome(chromo, run_number))
# #         run_number += 1
# #
# #     new_generation_chromosomes_pt3 = []
# #
# #     for chromo in new_generation_chromosomes_pt2:
# #         new_generation_chromosomes_pt3.append(mutate_chromosome(chromo, run_number))
# #         run_number += 1
# #
# #     new_generation_chromosomes = (new_generation_chromosomes_pt1 +
# #                                     new_generation_chromosomes_pt2 +
# #                                     new_generation_chromosomes_pt3)
# #
# #     return new_generation_chromosomes
#
# new_generation_chromosomes = create_full_new_generation(winner_chromosomes, last_run_from_previous_gen)
#
# # len(new_generation_chromosomes)
#
#
# # pprint.pprint(new_generation_chromosomes)
#
# file_path = '/Users/blakecuningham/Dropbox/MScDataScience/Thesis/EDEN/test_models/fixed_params'
# for chromo in new_generation_chromosomes:
#     chromo_file_name = chromo[0]
#     with open(file_path + chromo_file_name, "wb") as filehandle:
#         pickle.dump(chromo[1], filehandle)
#
#
# chromodate = new_generation_chromosomes[0][0].split('-')[0].split('chromo')[1]
# overallrunattempt = new_generation_chromosomes[0][0].split('N-')[1].split('-')[0].split('.')[0]
#
# list_file_name = ('list_pop_' + chromodate +
#                     '-N-' + str(overallrunattempt) + '.json')
#
# with open(file_path + list_file_name, 'w') as outfile:
#     json.dump(population_list, outfile)
#

# child_chromosome_1[1][1]

#TODO:
# do i only select winners once? (EDEN selects them multiple times)
# do i want to combine winners as true parents?
# why am i looping to find winners? can't i just take max?
# create genetic operators
# mutate winners (do i want to do EDEN method?)


# model0 = model_output_to_pdframe(population_of_models[0])


# print(tabulate(overall_results_frame_sorted))

# writer = ExcelWriter(model_list.replace('.json', '') + '.xlsx')
# overall_results_frame_sorted.to_excel(writer,'Sheet1')
# writer.save()

# print(overall_results_frame)

# population_of_models[0]

# # Testing learning rate distribution
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# # data = [((random.gauss(0, 1) / 1e4) + 1e-4) for _ in range(1000)]
#
# mu = 0
# sigma = 1
#
# data = [((random.lognormvariate(mu, sigma) / 1e4)) for _ in range(1000)]
# print(np.exp(mu + ((sigma^2)/2))/1e4)
# print(np.exp(mu)/1e4)
# plt.hist(data, bins=100)
# plt.axvline(x=1e-4, color='r', linestyle='dashed', linewidth=2)
# plt.show()

# random.choice([3,4,5,6])
# random.randrange(3,7)



#
# mutate_learn_rate(1)
#
# random.normalvariate(0, 1)
#
# random.lognormvariate(0, 1)
#
# np.exp(1)
#
# np.sqrt(np.square(np.e) - np.e)
#
# np.exp(0)/1e4

#
# import matplotlib.pyplot as plt
# # data = [((random.gauss(0, 1) / 1e4) + 1e-4) for _ in range(1000)]
#
# mu = 0
# sigma = 1
# scale = 1e4
#
# # data = [(generate_learn_rate() + (random.lognormvariate(mu, sigma)/1e4) - (np.exp(sigma)/1e4)) for _ in range(10000)]
# # data = [(generate_learn_rate() + (random.lognormvariate(mu, sigma)/scale) - (np.exp(sigma)/scale)) for _ in range(10000)]
# # data = [(generate_learn_rate() + (random.normalvariate(mu, sigma)/scale)) for _ in range(10000)]
# data = [mutate_learn_rate(1e-4) for _ in range(10000)]
# # data = [(generate_learn_rate()) for _ in range(10000)]
# # print(np.exp(mu + ((sigma^2)/2))/1e4)
# # print(np.exp(mu)/1e4)
# plt.hist(data, bins=100)
# plt.axvline(x=0, color='r', linestyle='dashed', linewidth=2)
# plt.show()
# #
# # random.choice([3,4,5,6])
# # random.randrange(3,7)
# Generation - this could just be a continuation of the N? Or do we need a sep gen number?


# winner_chromosomes[0] # Actual chromosome name and chromosome
# winner_chromosomes[0][0] # Chromosome name
# winner_chromosomes[0][1] # Actual chromosome
# winner_chromosomes[0][1][6] # Chromosome layers
# len(winner_chromosomes[0][1][6]) # Number of chromosome layers
# winner_chromosomes[0][1][6][0] # Specific chromosome layer

# type(winner_chromosomes)

# del winner_chromosomes[0][1][6][0]

# def mutute_chromosome(chromosome):
#     #TODO
#     # select layer
#     # change layer with random operator (delete, change, add)
#     # ensure model is valid
#     # add to population (I also need to add all the other stuff?)
#     return
#

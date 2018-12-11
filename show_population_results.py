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

from evolution_helpers import *

# Path variables

# user_path = '/Users/blakecuningham/'

# EDEN_dir = '../../EDEN/test_models/fixed_params/'
model_list = 'list_pop_2018127-N-1-G-0.json'
model_list_dir = 'models/' + model_list

# Functions

# test_model_name = 'chromo20181021-LR-75e-05-P-32128-run-5-N-0'

# def load_population_json(pop_list_path):
#
#     with open(model_list_dir) as json_data:
#         model_list_data = json.load(json_data)
#
#     return model_list_data
#
# def model_output_to_pdframe(model_tested):
#
#     model_name_for_directory = model_tested[0].replace('.p','').replace('.','')
#
#     ## set log directory
#
#     # log_directory_main = user_path + 'Documents/temp_RL_output/'
#     log_directory_main = '/Volumes/blakecjdl/RL_output/'
#     log_directory = log_directory_main + model_name_for_directory
#
#     train0_directory = log_directory.replace('~/','') + '/train_0'
#
#     for file in os.listdir(train0_directory):
#         log_file = file
#
#     eventsfile_path = train0_directory + '/' + log_file
#
#     acc = ea.EventAccumulator(eventsfile_path)
#     acc.Reload()
#
#     ep_reward = [(s.step, s.value) for s in acc.Scalars('global/episode_reward')]
#     ep_length = [(s.step, s.value) for s in acc.Scalars('global/episode_length')]
#     ep_time = [(s.step, s.value) for s in acc.Scalars('global/episode_time')]
#     reward_df = pd.DataFrame(ep_reward, columns = ('step', 'ep_reward'))
#     length_df = pd.DataFrame(ep_length, columns = ('step', 'ep_length'))
#     time_df = pd.DataFrame(ep_time, columns = ('step', 'ep_time'))
#     reward_df['EMA'] = reward_df['ep_reward'].ewm(alpha = 0.01).mean()
#     merged_df_1 = pd.merge(reward_df, length_df, on = ['step'])
#     merged_df_2 = pd.merge(merged_df_1, time_df, on = ['step'])
#
#     return merged_df_2
#
# def model_name(model_tested):
#
#     model_name_for_directory = model_tested[0].replace('.p','').replace('.','')
#
#     return model_name_for_directory

def model_results_summary(model_tested):

    name_of_model = model_name(model_tested)

    try:
        test_model_results_df =  model_output_to_pdframe(model_tested)
    except:
        test_model_results_df = pd.DataFrame({'step': [1],
                                                'ep_reward': [-21],
                                                'EMA': [-21],
                                                'ep_length': [1],
                                                'ep_time': [1]})

    ## The best episode in the experiment
    resultstable_bestepisode = test_model_results_df['ep_reward'].max()

    ## The number of times the result of the best episode was achieved
    resultstable_bestepisodecount = np.sum(test_model_results_df['ep_reward'] == test_model_results_df['ep_reward'].max())

    ## The best running score (excluding first 50 episodes to remove initial random bias)
    resultstable_bestEMA = test_model_results_df['EMA'][50:].max()

    ## Unbiased variance and standard deviation of series
    # test_model_results_df['ep_reward'].var()
    resultstable_overallSTD = test_model_results_df['ep_reward'].std()

    ## Time / steps to get to X level (acceleration)
    # try:
    #     steps_n200 = test_model_results_df[test_model_results_df['EMA'] >= -20.0][:1]['step'].values[0]
    # except:
    #     steps_n200 = np.nan

    try:
        steps_n195 = test_model_results_df[100:][test_model_results_df[100:]['EMA'] >= -20.4][:1]['step'].values[0]
    except:
        steps_n195 = np.nan

    ## Post X level stability

    try:
        steps_n195_index = test_model_results_df[test_model_results_df['step'] == steps_n195].index[0]
        resultstable_postn195SDT = test_model_results_df[steps_n195_index:]['ep_reward'].std()


        ## Model speed - average episode time for games above X score

        resultstable_postn195timeavg = test_model_results_df[steps_n195_index:]['ep_time'].mean()

        # Model efficiency - average steps for games above X score

        resultstable_postn195stepavg = test_model_results_df[steps_n195_index:]['ep_length'].mean()

    except:
        resultstable_postn195SDT = np.nan
        resultstable_postn195timeavg = np.nan
        resultstable_postn195stepavg = np.nan

    ## Model structure info

    conv_count = 0
    pool_count = 0
    dense_count = 0

    for layer in model_tested[1][6]:
        if layer['name'] == 'conv_2d':
            conv_count += 1
        elif layer['name'] == 'max_pool_2d':
            pool_count += 1
        elif layer['name'] == 'fully_conn':
            dense_count += 1

    ## Model end reason

    ## Noise around EMA

    noise_std = (test_model_results_df['ep_reward'] - test_model_results_df['EMA']).std()

    ## General speed metric (MA distance / step)

    # np.max((test_model_results_df['EMA'][100:] + 21) / test_model_results_df['step'][100:])

    resultstable_vector = OrderedDict()
    # resultstable_vector['Model'] = [str(name_of_model)]
    resultstable_vector['LR'] = model_tested[1][1]
    resultstable_vector['Params'] = model_tested[1][2][0]
    resultstable_vector['Image net output'] = model_tested[1][2][1]
    resultstable_vector['Conv_2d layers'] = conv_count
    resultstable_vector['Max_pool_2d layers'] = pool_count
    resultstable_vector['Fully_conn layers'] = dense_count
    resultstable_vector['Best episode'] = [int(resultstable_bestepisode)]
    resultstable_vector['Count best episodes'] = [int(resultstable_bestepisodecount)]
    resultstable_vector['Best EMA'] = [round(resultstable_bestEMA,2)]
    resultstable_vector['Overall StD'] = [round(resultstable_overallSTD,2)]
    resultstable_vector['Noise around EMA'] = [round(noise_std, 2)]
    resultstable_vector['Total steps'] = test_model_results_df['step'].tail(1)
    resultstable_vector['Total time (s)'] = round(np.sum(test_model_results_df['ep_time']),0)
    resultstable_vector['Avg. episode steps'] = test_model_results_df['ep_length'].mean()
    resultstable_vector['Avg. episode time'] = test_model_results_df['ep_time'].mean()
    resultstable_vector['Avg. time per step'] = (test_model_results_df['ep_time'].sum() / test_model_results_df['ep_length'].sum())
    resultstable_vector['Steps to mastery'] = [steps_n195]
    resultstable_vector['Post mastery StD'] = [round(resultstable_postn195SDT,2)]
    resultstable_vector['Post mastery avg. episode steps'] = [round(resultstable_postn195stepavg,2)]
    resultstable_vector['Post mastery avg. episode time'] = [round(resultstable_postn195timeavg,2)]

    return pd.DataFrame(resultstable_vector)


population_of_models = load_population_json(model_list_dir)

# TODO: Add fail reason
# TODO: Add mastery level
# model_results_summary(population_of_models[3])

overall_results_frame = pd.DataFrame()

for model in population_of_models:
    overall_results_frame = overall_results_frame.append(model_results_summary(model), ignore_index=True)

overall_results_frame_sorted = overall_results_frame.sort_values(['Best EMA'], ascending=[False])


print(tabulate(overall_results_frame_sorted))

writer = ExcelWriter(model_list.replace('.json', '') + '.xlsx')
overall_results_frame_sorted.to_excel(writer,'Sheet1')
writer.save()

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

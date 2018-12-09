import tensorflow as tf
import pandas as pd
import os
import json
from tensorboard.backend.event_processing import event_accumulator as ea
import subprocess
import time
import numpy as np

# import re

# Other options

enforce_step_limit = True
enforce_time_limit = True

experiment_time_limit_secs = 3600
experiment_step_limit = 16000000

error_time_limit_secs = 300


# Path variables

# user_path = '/Users/blakecuningham/'

# EDEN_dir = "../../EDEN/test_models/"

log_directory = 'output/'
model_list = "list_pop_2018127-N-1-G-0.json"

# Functions

def current_running_reward(log_directory):

    # user_path = '/Users/blakecuningham/'
    train0_directory = log_directory + '/train_0'
    train0_directory = train0_directory.replace('~/','')
    full_train0log_directory = user_path + train0_directory
    full_train0log_directory = train0_directory

    for file in os.listdir(full_train0log_directory):
        log_file = file

    eventsfile_path = full_train0log_directory + '/' + log_file

    acc = ea.EventAccumulator(eventsfile_path)
    acc.Reload()

    ep_reward = [(s.step, s.value) for s in acc.Scalars('global/episode_reward')]
    reward_df = pd.DataFrame(ep_reward[-500:], columns = ("step", "ep_reward"))

    fail21 = (np.sum(reward_df[-50:]["ep_reward"] == -21) == 50)

    # alpha of 0.01 is the same as the original Karpathy metric - weights the new score as 1% towards the MA
    return [round(reward_df["ep_reward"].ewm(alpha = 0.01).mean().tail(1).item(), 2), reward_df["step"].tail(1).item(), fail21]

def best_running_reward(log_directory):

    # user_path = '/Users/blakecuningham/'
    train0_directory = log_directory + '/train_0'
    train0_directory = train0_directory.replace('~/','')
    full_train0log_directory = user_path + train0_directory
    full_train0log_directory = train0_directory

    for file in os.listdir(full_train0log_directory):
        log_file = file

    eventsfile_path = full_train0log_directory + '/' + log_file

    acc = ea.EventAccumulator(eventsfile_path)
    acc.Reload()

    ep_reward = [(s.step, s.value) for s in acc.Scalars('global/episode_reward')]
    reward_df = pd.DataFrame(ep_reward, columns = ("step", "ep_reward"))
    # alpha of 0.01 is the same as the original Karpathy metric - weights the new score as 1% towards the MA
    return round(reward_df["ep_reward"].ewm(alpha = 0.01).mean()[50:].max(), 2)

def run_experiment_chromosome_mod(model_for_experiment, log_directory_main):

    model_name_for_directory = model_for_experiment[0].replace('.p','').replace('.','')

    ## set log directory

    # log_directory_main = user_path + 'Documents/temp_RL_output/'
    log_directory = log_directory_main + model_name_for_directory

    ## set model directory

    model_path = EDEN_dir + model_for_experiment[0]

    ## begin experiment

    process_start_command = ['python', 'train.py',
                                '--num-workers', '2',
                                '--env-id', 'PongDeterministic-v3',
                                '--log-dir', log_directory,
                                '--model-dir', model_path]

    subprocess.call(process_start_command)

    print("\n\n\n")
    print("Running model: " + model_name_for_directory)
    print("\n\n\n")

    end_experiment = False
    start_time = time.time()

    while not end_experiment:
        time.sleep(60)
        elapsed_time = time.time() - start_time
        try:
            print("Current score: " + str(current_running_reward(log_directory)[0]) + ", Time : " + str(round(elapsed_time,2)) + ", Step: " + str(round(current_running_reward(log_directory)[1])))
            if (enforce_time_limit) and (elapsed_time > experiment_time_limit_secs):
                end_experiment = True
                print("\n")
                print("ENDING experiment MAX TIME - score:" + str(current_running_reward(log_directory)[0]))
                print("\nBest reward: " + str(round(best_running_reward(log_directory),2)))
                print("\n")
            elif (current_running_reward(log_directory)[0] < -20.99) and (elapsed_time > 1200):
                end_experiment = True
                print("\n")
                print("ENDING experiment FAILED (running score too low)- score:" + str(current_running_reward(log_directory)[0]))
                print("\nBest reward: " + str(round(best_running_reward(log_directory),2)))
                print("\n")
            elif (current_running_reward(log_directory)[2]) and (elapsed_time > 300):
                end_experiment = True
                print("\n")
                print("ENDING experiment FAILED (too many -21 games in a row) - score:" + str(current_running_reward(log_directory)[0]))
                print("\nBest reward: " + str(round(best_running_reward(log_directory),2)))
                print("\n")
            elif (enforce_step_limit) and (current_running_reward(log_directory)[1] > experiment_step_limit):
                end_experiment = True
                print("\n")
                print("ENDING experiment MAX STEPS - score:" + str(current_running_reward(log_directory)[0]))
                print("\nBest reward: " + str(round(best_running_reward(log_directory),2)))
                print("\n")
        except:
            if (elapsed_time > error_time_limit_secs):
                end_experiment = True
                print("Error time limit reached")
            else:
                print("Potential time out - sleeping for 1 min")
                time.sleep(60)
    # Kill experiment when conditions satisfied

    subprocess.call(['tmux','kill-session', '-t', 'a3c'])
    time.sleep(10)


# current_running_reward(log_directory)[0]

# Read in list of models (generation 1)

model_list_dir = "models/" + model_list

with open(model_list_dir) as json_data:
    model_list_data = json.load(json_data)

# Loop through experiments for models

# Normal
for mod in model_list_data:
    run_experiment_chromosome_mod(mod, log_directory)


# Evaluate and create Create new generation

# Run new generation for model in new generation




###################
# # Interupted
###################
# offset = 5
# for i in range(len(model_list_data) - offset):
#     run_experiment_chromosome_mod(model_list_data[i + offset])

# Run OpenAi model Only

# run_experiment_chromosome_mod(model_list_data[11])

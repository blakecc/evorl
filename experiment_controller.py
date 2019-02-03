import subprocess
import time
import json
from evolution_helpers import *

# Common variables

tmux_session_name = '181223_experiment'
pem_file = '~/aws/rl_learner.pem'
local_model_folder = '/Users/blakecuningham/Documents/evorl/models'
local_output_folder = '/Users/blakecuningham/Documents/evorl/output'

model_list_file = 'list_pop_20181219-N-0-G-8.json'

list_of_instances = ['ubuntu@ec2-34-254-159-143.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-34-251-169-103.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-34-254-63-146.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-52-209-92-98.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-18-203-185-83.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-34-255-160-2.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-54-194-111-138.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-34-243-129-73.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-52-215-188-208.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-34-244-218-246.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-63-33-197-22.eu-west-1.compute.amazonaws.com',
                        'ubuntu@ec2-63-33-63-241.eu-west-1.compute.amazonaws.com']

# list_of_instances = ['ubuntu@ec2-34-254-159-143.eu-west-1.compute.amazonaws.com',
#                         'ubuntu@ec2-34-251-169-103.eu-west-1.compute.amazonaws.com',
#                         'ubuntu@ec2-34-254-63-146.eu-west-1.compute.amazonaws.com']


## Initial setup
# 1. ensure all python files neccessary on instance

## for the generation

def specific_model_run_start(pem_file, instance_address, model_number_to_run, tmux_session_name, model_list_data, model_list_file, local_model_folder):

    ## Individual experiment process
    # 0. Upload run_experiment_static.py

    subprocess.run(['scp','-r','-i', pem_file, '/Users/blakecuningham/Documents/evorl' + '/' + 'run_experiment_static.py', instance_address + ':~/evorl'])

    # 1. upload model list

    subprocess.run(['scp','-r','-i', pem_file, local_model_folder + '/' + model_list_file, instance_address + ':~/evorl/models'])

    # 2. upload model

    subprocess.run(['scp','-r','-i', pem_file, local_model_folder + '/' + model_list_data[model_number_to_run][0], instance_address + ':~/evorl/models'])

    # 3. kill all sessions

    subprocess.run(['ssh', '-t', '-i', pem_file, instance_address, 'tmux kill-server'])
    subprocess.run(['ssh', '-t', '-i', pem_file, instance_address, 'rm -r ~/evorl/__pycache__'])

    # 4. create a tmux session

    subprocess.run(['ssh', '-t', '-i', pem_file, instance_address, 'tmux new -d -s ' + tmux_session_name])

    # 5. run experiment for specific model
    # 5a ensure right directory

    subprocess.run(['ssh', '-t', '-i', pem_file, instance_address, 'tmux send-keys -t ' + tmux_session_name + " \'cd ~/evorl\' Enter"])

    # 5b activate python env

    subprocess.run(['ssh', '-t', '-i', pem_file, instance_address, 'tmux send-keys -t ' + tmux_session_name + " \'source activate tensorflow_p36\' Enter"])

    # 5c run python file

    subprocess.run(['ssh', '-t', '-i', pem_file, instance_address, 'tmux send-keys -t ' + tmux_session_name + " \'python run_experiment_static.py -l \'" + model_list_file + "\' -n " + str(model_number_to_run) + "\' Enter"])

    print('Successfully started model number ' + str(model_number_to_run) + '.\n')

    return 0

def specific_model_run_end(pem_file, instance_address, model_number_to_run, tmux_session_name, model_list_data, model_list_file, local_model_folder):

    # time.sleep(30)
    subprocess.run(['ssh', '-t', '-i', pem_file, instance_address, 'tmux kill-server'])

    # 7. download model results
    current_model_name = model_list_data[model_number_to_run][0].replace('.p','').replace('.','')
    subprocess.run(['scp','-r','-i', pem_file, instance_address + ':~/evorl/output/' + current_model_name + '/', local_output_folder])

    print('Successfully ended and transferred model number ' + str(model_number_to_run) + '.\n')

    return 0

def generation_run(model_list_file, local_model_folder, pem_file, tmux_session_name, list_of_instances):

    model_list_dir = local_model_folder + '/' + model_list_file

    num_models = len(list_of_instances)

    with open(model_list_dir) as json_data:
        model_list_data = json.load(json_data)

    for mod in range(num_models):
        instance_address = list_of_instances[mod]
        specific_model_run_start(pem_file, instance_address, mod, tmux_session_name, model_list_data, model_list_file, local_model_folder)

    print('Successfully started all models in ' + model_list_file + '.\n')

    for hr in range(12):
        print('So far, ' + str(hr) + 'hrs elapsed.\n')
        time.sleep(60*60)
        # time.sleep(60)

    print('Starting 10 minute settle down period.\n')
    time.sleep(10*60)
    # time.sleep(30)

    for mod in range(num_models):
        instance_address = list_of_instances[mod]
        specific_model_run_end(pem_file, instance_address, mod, tmux_session_name, model_list_data, model_list_file, local_model_folder)

    print('Successfully ended all models in ' + model_list_file + '.\n')

    return 0

def generation_run_end(model_list_file, local_model_folder, pem_file, tmux_session_name, list_of_instances):

    model_list_dir = local_model_folder + '/' + model_list_file

    num_models = len(list_of_instances)

    with open(model_list_dir) as json_data:
        model_list_data = json.load(json_data)

    # for mod in range(num_models):
    #     instance_address = list_of_instances[mod]
    #     specific_model_run_start(pem_file, instance_address, mod, tmux_session_name, model_list_data, model_list_file, local_model_folder)
    #
    # print('Successfully started all models in ' + model_list_file + '.\n')
    #
    # for hr in range(12):
    #     print('So far, ' + str(hr) + 'hrs elapsed.\n')
    #     time.sleep(60*60)
    #     # time.sleep(60)
    #
    # print('Starting 10 minute settle down period.\n')
    # time.sleep(10*60)
    # # time.sleep(30)

    for mod in range(num_models):
        instance_address = list_of_instances[mod]
        specific_model_run_end(pem_file, instance_address, mod, tmux_session_name, model_list_data, model_list_file, local_model_folder)

    print('Successfully ended all models in ' + model_list_file + '.\n')

    return 0

## Uncomment below when starting right from beginning and wanting to run gen 0
# generation_run(model_list_file, local_model_folder, pem_file, tmux_session_name, list_of_instances)

## Between generations
# 1. run new_generation script for N generation

## Using below to redo ending of a generation after a transfer of files stalled
generation_run_end(model_list_file, local_model_folder, pem_file, tmux_session_name, list_of_instances)

## Changed this to run the last two generations - need to fix gen + n
for gen in range(2):
    print('About to generate generation ' + str(gen + 9) + '.\n')
    model_list_file =  new_generation_main(model_list_file, gen + 1)
    generation_run(model_list_file, local_model_folder, pem_file, tmux_session_name, list_of_instances)

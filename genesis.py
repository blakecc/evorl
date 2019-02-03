import pandas as pd
import numpy as np
import time
import sys
import pickle
import json

from evolution_helpers import *

## Set key harcoded parameters ##

run_attempt = 0
first_generation_size = 11

# The seed which will be used by Python random and scikit learn Kfold
seed_value = 7

# Run
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                                    MAIN LOOP
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
population_list = []
# file_path = '/Users/blakecuningham/Dropbox/MScDataScience/Thesis/EDEN/test_models/fixed_params'
file_path = "models/"

chromo = generate_openai_chromosome(4)

chromo_file_name = ('chromo' + str(dt.date.today().year) +
                    str(dt.date.today().month) +
                    str(dt.date.today().day) +
                    '-LR-' + str(chromo[1]) +
                    '-P-' + str(chromo[2][0]) + '-O-' + str(chromo[2][1]) +
                    '-run-' + 'A' + '-N-' + str(run_attempt) + '.p')

print_chromosome_details(chromo)
get_model_keras_summary(chromo[6])

population_list.append([chromo_file_name, chromo])

with open(file_path + chromo_file_name, "wb") as filehandle:
    pickle.dump(chromo, filehandle)

for run in range(first_generation_size):

    # Choose chromosome size
    chromosome_size = random.choice([3,4,5,6])

    chromo = generate_one_chromosome_specified_size(chromosome_size)

    chromo_file_name = ('chromo' + str(dt.date.today().year) +
                        str(dt.date.today().month) +
                        str(dt.date.today().day) +
                        '-LR-' + str(chromo[1]) +
                        '-P-' + str(chromo[2][0]) + '-O-' + str(chromo[2][1]) +
                        '-run-' + str(run) + '-N-' + str(run_attempt) + '.p')

    print_chromosome_details(chromo)
    get_model_keras_summary(chromo[6])

    population_list.append([chromo_file_name, chromo])

    with open(file_path + chromo_file_name, "wb") as filehandle:
        pickle.dump(chromo, filehandle)

list_file_name = ('list_pop_' + str(dt.date.today().year) +
                    str(dt.date.today().month) +
                    str(dt.date.today().day) +
                    '-N-' + str(run_attempt) + '-G-0.json')

with open(file_path + list_file_name, 'w') as outfile:
    json.dump(population_list, outfile)

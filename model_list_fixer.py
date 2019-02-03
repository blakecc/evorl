# I accidentally saved new model lsit files with the wrong generation number
# and wrote over previous versions. Using this to try fix them (gen 1 and 2)

import pickle
import os
import json


###############################################################################
####### Gen 1 #################################################################
###############################################################################

list_of_gen_1_models = []

for f in os.listdir('models'):
    if f.endswith('.p'):
        try:
            run_num = int(f.split('-run-')[1].split('-')[0])
            if run_num >= 11 and run_num <= 22:
                list_of_gen_1_models.append([f, run_num])
        except:
            run_num = f.split('-run-')[1].split('-')[0]
            print("Exception with run: " + run_num + '\n')

# len(list_of_gen_1_models)

def getKey(item):
    return item[1]

list_of_gen_1_models = sorted(list_of_gen_1_models, key=getKey)

generation_of_models_list_1 = []

for mod in list_of_gen_1_models:
    mod_path = 'models/' + mod[0]

    with open(mod_path, "rb") as filehandle:
        chromo_detail = pickle.load(filehandle)

    generation_of_models_list_1.append([mod[0], chromo_detail])

with open('models/' + 'list_pop_20181219-N-0-G-1.json', 'w') as outfile:
    json.dump(generation_of_models_list_1, outfile)

###############################################################################
####### Gen 2 #################################################################
###############################################################################

list_of_gen_2_models = []

for f in os.listdir('models'):
    if f.endswith('.p'):
        try:
            run_num = int(f.split('-run-')[1].split('-')[0])
            if run_num >= 23 and run_num <= 34:
                list_of_gen_2_models.append([f, run_num])
        except:
            run_num = f.split('-run-')[1].split('-')[0]
            print("Exception with run: " + run_num + '\n')

# len(list_of_gen_2_models)

list_of_gen_2_models = sorted(list_of_gen_2_models, key=getKey)

generation_of_models_list_2 = []

for mod in list_of_gen_2_models:
    mod_path = 'models/' + mod[0]

    with open(mod_path, "rb") as filehandle:
        chromo_detail = pickle.load(filehandle)

    generation_of_models_list_2.append([mod[0], chromo_detail])

with open('models/' + 'list_pop_20181219-N-0-G-2.json', 'w') as outfile:
    json.dump(generation_of_models_list_2, outfile)

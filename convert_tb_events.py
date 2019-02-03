import json
import os
import pandas as pd
import sys, traceback
from evolution_helpers import *

json_list = []

for mod in os.listdir('models'):
    if mod.endswith('.json'):
        list_num = int(mod.split('-G-')[1].split('.json')[0])
        json_list.append([mod, list_num])

def getKey(item):
    return item[1]

json_list = sorted(json_list, key=getKey)

all_model_list_data = []

for modlist in json_list:
    all_model_list_data.append(load_population_json('models/' + modlist[0]))

model_table = []

gen_num = 0

for gen in all_model_list_data:
    for mod in gen:
        try:
            model_output_to_pdframe(mod).to_csv(
                'csv_output/' + mod[0].replace('p','').replace('.','') + '.csv', index = False)
            model_table.append([mod[0].replace('p','').replace('.',''), str(gen_num)])
            print('Successfully converted: ' + mod[0] + '\n')
        except:
            print('Failed to convert: ' + mod[0] + '\n')
            traceback.print_exc(file=sys.stdout)
    gen_num += 1


# model_table
# all_model_list_data

pd.DataFrame(model_table, columns = ['Model', 'Generation']).to_csv('csv_output/list_of_models_tested.csv', index = False)

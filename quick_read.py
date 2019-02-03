import json
import os
import pandas as pd
from evolution_helpers import *

# def load_population_json(pop_list_path):
#
#     with open(pop_list_path) as json_data:
#         model_list_data = json.load(json_data)
#
#     return model_list_data

json_list = []

for mod in os.listdir('models'):
    if mod.endswith('.json'):
        # print(mod)
        # json_list.append(load_population_json('models/' + mod))
        list_num = int(mod.split('-G-')[1].split('.json')[0])
        json_list.append([mod, list_num])

def getKey(item):
    return item[1]

json_list = sorted(json_list, key=getKey)


all_model_list_data = []

for modlist in json_list:
    all_model_list_data.append(load_population_json('models/' + modlist[0]))


# json_list[0][0][0].split('chromo20181219-')[1].split('-run')[0]

# json_list
# all_model_list_data

# model_table = pd.DataFrame(columns = ['Generation', 'Model'])
model_table = []

gen = 0
for group in all_model_list_data:
    # print("Gen: " + str(gen))
    for mod in all_model_list_data[gen]:
        mod_results = model_results_summary_for_tournament(mod)
        model_table.append([gen, mod[0].split('chromo20181219-')[1].split('-run')[0],
            mod_results['Best EMA'].values[0],
            mod_results['Noise around EMA'].values[0],
            mod_results['Avg. episode time'].values[0],
            mod_results['Tournament score'].values[0]])
        print('Successfully added model: ' + mod[0] + '\n')
    gen += 1

# model_results_summary_for_tournament(all_model_list_data[0][0])['Best EMA'].values[0]

# model_table

# pd.DataFrame(model_table, columns = ['Generation', 'Model'])

# writer = pd.ExcelWriter('181229_evolution.xlsx')
# pd.DataFrame(model_table, columns = ['Generation', 'Model', 'Best EMA', 'Noise around EMA', 'Avg. episode time', 'Tournament score']).to_excel(writer,'list')
# writer.save()

#### Other Latex tables

model_table

model_table_df = pd.DataFrame(model_table, columns = ['Generation', 'Model', 'Best EMA', 'Noise', 'EpTime', 'Score'])

model_table_df['Parameters'] = model_table_df['Model'].split('-P-')[1].split('-O-')[0]

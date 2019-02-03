import json
import os
import pandas as pd
from collections import OrderedDict
from evolution_helpers import *
from keras.utils import plot_model
import pprint
import numpy as np
from tabulate import tabulate

def add_conv2d_layer(mod_architecture_table, layer):
    mod_architecture_table = mod_architecture_table.append({'Type': layer['name'],
                                    'Size': layer['filter_size'],
                                    'Number': layer['nb_filter'],
                                    'Activation': layer['activation']}, ignore_index = True)
    return mod_architecture_table

def add_maxpool2d_layer(mod_architecture_table, layer):
    mod_architecture_table = mod_architecture_table.append({'Type': layer['name'],
                                    'Size': layer['kernel_size'],
                                    'Number': "",
                                    'Activation': ""}, ignore_index = True)
    return mod_architecture_table

def add_fullyconn_layer(mod_architecture_table, layer):
    mod_architecture_table = mod_architecture_table.append({'Type': layer['name'],
                                    'Size': "",
                                    'Number': layer['units'],
                                    'Activation': layer['activation']}, ignore_index = True)
    return mod_architecture_table

def add_layer_df(mod_architecture_table, layer):
    if layer['name'] == 'conv_2d':
        mod_architecture_table = add_conv2d_layer(mod_architecture_table, layer)
    elif layer['name'] == 'max_pool_2d':
        mod_architecture_table = add_maxpool2d_layer(mod_architecture_table, layer)
    elif layer['name'] == 'fully_conn':
        mod_architecture_table = add_fullyconn_layer(mod_architecture_table, layer)

    return mod_architecture_table

def model_layer_df(model):

    current_mod_architecture = model[1][6]
    mod_architecture_table = pd.DataFrame(columns = ['Type', 'Size', 'Number', 'Activation'])

    for layer in current_mod_architecture:
        # text_file.write(layer['name'])
        mod_architecture_table = add_layer_df(mod_architecture_table, layer)

    return mod_architecture_table

def print_model_details(current_model):

    text_file.write("\subsubsection*{" + current_model[0].replace('p','').replace('.','').split('chromo20181219-')[1].split('-run-')[0] + '}\n')

    # learning rate
    text_file.write("Learning rate: {:.6f}".format(round(current_model[1][1],6)) + '\n')

    # midlayer parameters
    text_file.write("\\\\Parameters in image layer: {:.0f}".format(current_model[1][2][0]) + '\n')

    # midlayer output vector
    text_file.write("\\\\Size of output from image layer to LSTM: {:.0f}".format(current_model[1][2][1]) + '\n')

    # layers

    conv_count = 0
    pool_count = 0
    dense_count = 0

    for layer in current_model[1][6]:
        if layer['name'] == 'conv_2d':
            conv_count += 1
        elif layer['name'] == 'max_pool_2d':
            pool_count += 1
        elif layer['name'] == 'fully_conn':
            dense_count += 1

    text_file.write("\\\\\\\\\\underline{Count of layers} \n")
    text_file.write("\\\\Conv 2D:           {:.0f}".format(conv_count))
    text_file.write("\\\\Max pool 2D:      {:.0f}".format(pool_count))
    text_file.write("\\\\Fully connected:  {:.0f}".format(dense_count))
    text_file.write('\n')

    model_layer_dataframe = model_layer_df(current_model)

    text_file.write("\\\\\\\\Detailed layers of model: \\\\")
    text_file.write(tabulate(model_layer_dataframe, headers = 'keys', tablefmt = 'latex'))
    # text_file.write(model_layer_dataframe_temp)
    # text_file.write('\n--------------------------------------------------')
    # text_file.write('--------------------------------------------------\n')

# def load_population_json(pop_list_path):
#
#     with open(pop_list_path) as json_data:
#         model_list_data = json.load(json_data)
#
#     return model_list_data


def add_model_details(current_model, gen_num, moddf):

    modvector = OrderedDict()

    modvector['gen'] = [gen_num]

    modvector['name'] = [current_model[0].replace('p','').split('chromo20181219-')[1].split('-run-')[0]]

    # learning rate
    modvector['learnrate'] = [round(current_model[1][1],6)]

    # midlayer parameters
    modvector['midparams'] = [current_model[1][2][0]]

    # midlayer output vector
    modvector['outnum'] = [current_model[1][2][1]]

    # layers

    conv_count = 0
    pool_count = 0
    dense_count = 0

    relu_count = 0
    elu_count = 0
    leakyrelu_count = 0

    number_conv_filters = 0
    filter_ratio = 1

    for layer in current_model[1][6]:
        if layer['name'] == 'conv_2d':
            conv_count += 1
            if layer['activation'] == 'elu':
                elu_count += 1
            elif layer['activation'] == 'relu':
                relu_count += 1
            elif layer['activation'] == 'leaky_relu':
                leakyrelu_count += 1
            current_layer_nb_filters = layer['nb_filter']
            number_conv_filters += current_layer_nb_filters
            if conv_count >= 2:
                filter_ratio = filter_ratio * (previous_layer_nb_filters / current_layer_nb_filters)
            previous_layer_nb_filters = layer['nb_filter']
        elif layer['name'] == 'max_pool_2d':
            pool_count += 1
        elif layer['name'] == 'fully_conn':
            dense_count += 1


    # text_file.write("\\\\\\\\\\underline{Count of layers} \n")
    modvector['count_conv2d'] = [conv_count]
    modvector['count_pool2d'] = [pool_count]
    modvector['count_fullyconn'] = [dense_count]
    modvector['count_elu'] = [elu_count]
    modvector['count_relu'] = [relu_count]
    modvector['count_leakyrelu'] = [leakyrelu_count]
    modvector['filter_ratio'] = [filter_ratio]
    modvector['number_conv_filters'] = [number_conv_filters]
    # text_file.write('\n')

    # model_layer_dataframe = model_layer_df(current_model)

    # text_file.write("\\\\\\\\Detailed layers of model: \\\\")
    # text_file.write(tabulate(model_layer_dataframe, headers = 'keys', tablefmt = 'latex'))
    # text_file.write(model_layer_dataframe_temp)
    # text_file.write('\n--------------------------------------------------')
    # text_file.write('--------------------------------------------------\n')

    updated_moddf = moddf.append(pd.DataFrame(modvector), ignore_index = True)

    return updated_moddf

# def load_population_json(pop_list_path):
#
#     with open(pop_list_path) as json_data:
#         model_list_data = json.load(json_data)
#
#     return model_list_data

json_list = []

for mod in os.listdir('models'):
    if mod.endswith('.json'):
        # text_file.write(mod)
        # json_list.append(load_population_json('models/' + mod))
        list_num = int(mod.split('-G-')[1].split('.json')[0])
        json_list.append([mod, list_num])

def getKey(item):
    return item[1]

json_list = sorted(json_list, key=getKey)

all_model_list_data = []

for modlist in json_list:
    all_model_list_data.append(load_population_json('models/' + modlist[0]))


# current_model = all_model_list_data[1][11]

# mod generation

# mod name

text_file = open("appendix_models.tex", "w")
text_file.write("\\section{List of models}\n")

model_details_dataframe = pd.DataFrame()

gen_num = 0
for gen in all_model_list_data:
    # text_file.write("\n##################################################")
    # text_file.write("##################################################\n")
    # text_file.write("Generation: " + str(gen_num) + '\n')
    # text_file.write("##################################################")
    # text_file.write("##################################################\n")
    text_file.write("\\subsection{Generation " + str(gen_num) + '}\n')
    for mod in gen:
        print_model_details(mod)
        model_details_dataframe = add_model_details(mod, gen_num, model_details_dataframe)
    gen_num += 1

text_file.close()

model_details_dataframe.to_excel('190103_ModelDetailsSummary.xlsx')

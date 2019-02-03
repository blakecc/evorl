import os
# import subprocess
import shutil
# https://stackoverflow.com/questions/6996603/delete-a-file-or-folder

for item_0 in os.listdir('output'):
    # print('output/' + item_0)
    if os.path.isdir('output/' + item_0):
        for item_1 in os.listdir('output/' + item_0):
            if item_1 == 'train_1' or item_1 == 'train':
                print('Attempting to delete: ' + item_0 + '/' + item_1)
                shutil.rmtree('output/' + item_0 + '/' + item_1)

print('\nSuccessfully deleted all redundant files.\nProbably.\n')

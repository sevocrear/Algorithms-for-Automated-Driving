import os
import shutil
def mkdir_if_not_exist(file):
    try:
        os.mkdir(file)
    except FileExistsError:
        x = input('Folder for images exists already. Do you want to delete folder and create again?type y or n:\n')
        x = x.lower()
        if x == 'y':
            shutil.rmtree(file)
            os.mkdir(file)
        pass
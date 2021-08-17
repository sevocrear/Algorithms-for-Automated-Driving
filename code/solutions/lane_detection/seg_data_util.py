import os
def mkdir_if_not_exist(file):
    try:
        os.mkdir(file)
    except FileExistsError:
        pass
import os
import pathlib
    
def get_data_path():
    path = os.environ.get('MINIMINT_DATA_PATH')
    if path is not None:
        return path
    path = str(pathlib.Path(__file__).parent.absolute()) +'/data/'
    os.makedirs(path, exist_ok=True)
    return path

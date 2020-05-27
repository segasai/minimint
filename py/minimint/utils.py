import os
import pathlib
import tempfile

def get_data_path():
    path = os.environ.get('MINIMINT_DATA_PATH')
    if path is not None:
        return path
    path = str(pathlib.Path(__file__).parent.absolute()) + '/data/'
    os.makedirs(path, exist_ok=True)
    return path


def tail_head(fin, nskip, nout):
    """ 
    Read nout lines from fin after skipping nskip lines 
    and put output in the temporary file. Return filename
    """
    fp = open(fin, 'r')
    fpout = tempfile.NamedTemporaryFile(delete=False, mode='w')
    i = -1
    for l in fp:
        i += 1
        if i < nskip:
            continue
        print(l, file=fpout)
        if i == (nskip + nout):
            break
    fp.close()
    fpout.close()
    return fpout.name

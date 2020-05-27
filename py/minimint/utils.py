import os
import pathlib


def get_data_path():
    path = os.environ.get('MINIMINT_DATA_PATH')
    if path is not None:
        return path
    path = str(pathlib.Path(__file__).parent.absolute()) + '/data/'
    os.makedirs(path, exist_ok=True)
    return path


def tail_head(fin, fout, nskip, nout):
    # read nout lines from fin after skipping nskip lines and put output in fout
    fp = open(fin, 'r')
    fpout = open(fout, 'w')
    i = -1
    for l in fp:
        i += 1
        if i < nskip:
            continue
        print(fpout, file=fpout)
        if i == (nskip + nout):
            break
    fp.close()
    fpout.close()

import nrrd
import numpy as np

if __name__ == '__main__':
    switch = 'to_npy'

    if switch == 'to_nrrd':
        npy_path = 'P1_2_K0_cropped.npy'
        nrrd_path = npy_path.split('.')[0]+'.nrrd'
        base_dir = '../results/graph_cut_res/'
        nrrd.write(base_dir+nrrd_path, np.load(base_dir+npy_path).astype(np.float))
    else:
        nrrd_path = 'essence_artery_kidney.nrrd'
        npy_path = nrrd_path.split('.')[0]+'.npy'
        # base_dir = '../nrrd/cropped/'
        base_dir = '../results/secondcase/'
        np.save(base_dir + npy_path, nrrd.read(base_dir + nrrd_path)[0])

import nrrd
import numpy as np
import torch
import cv2
from skimage.morphology import square, disk, binary_erosion, label
from scipy.ndimage.morphology import binary_fill_holes

import pickle

with open("results/secondcase/mid_res/mid_res_essence_kidney", "rb") as fp:
    res_dict = pickle.load(fp)
    index_list = res_dict['index']
    cca_mask = res_dict['mask']

all_mask = np.zeros(cca_mask.shape)
if len(index_list) > 1:
    sorted_ind = sorted(index_list, key=lambda x: x[1], reverse=True)
    # selected_inds = [sorted_ind[1][0],sorted_ind[2][0]]
    selected_inds = [sorted_ind[1], sorted_ind[2], sorted_ind[3]]  # seolen

    for j in selected_inds:
        inds = np.where(cca_mask == j[0])
        shape = inds[0].shape
        print(shape)
        all_mask[inds] = 1
    fill_mask = binary_fill_holes(all_mask)
nrrd.write('results/secondcase/essence_kidney_3part.nrrd', fill_mask.astype(np.float))

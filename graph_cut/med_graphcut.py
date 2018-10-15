import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

from medpy.graphcut.energy_voxel import boundary_difference_exponential, regional_probability_map
from medpy.graphcut import graph_from_voxels

import nrrd
from utils import *
import argparse

def med_graphcut(img, alpha, sigma):
    # to be initialized
    '''
    :alpha: balancing between boundary and regional term.
    :sigma: use in the boundary term. sigma^2 = 1/beta.
    : !spacing: A sequence containing the slice spacing used for weighting
        the computed neighbourhood weight value for different dimensions
    :param img:
    :return:
    '''
    # sigma, alpha = 40. , 1e-2
    spacing = False

    # fg_thresh, bg_thresh = 200, 1  # values between them are Undetermined.

    # set foreground, background seeds
    fg_mark, bg_mark = np.zeros_like(img), np.zeros_like(img)
    img[img<0] = 0      # ignore negative intensity

    # only for arterial_essence suppression
    max_essence_intensity = 320
    img[img>max_essence_intensity] = max_essence_intensity

    # fg_mark[img>=fg_thresh] = 1
    # bg_mark[img<=bg_thresh] = 1

    # probability map
    # A defect: the histogram should not be the whole image, but the foreground distribution.
    # probability_map = prob_map(img, prob_histogram(img, (bg_thresh+1, 255)))  # bins=256 or others
    # probability_map = np.zeros(img.shape)+1e-4  # only use boundary term
    probability_map = prob_map_md(img)
    # print(probability_map)

    gcgraph = graph_from_voxels(fg_mark,
                                bg_mark,
                                regional_term= regional_probability_map,
                                regional_term_args= (probability_map, alpha),
                                boundary_term = boundary_difference_exponential,
                                boundary_term_args = (img, sigma, spacing))

    maxflow = gcgraph.maxflow()

    result_mask = np.zeros(img.size, dtype=np.bool)
    for idx in range(len(result_mask)):
        result_mask[idx] = 0 if gcgraph.termtype.SINK == gcgraph.what_segment(idx) else 1  # ?
    result_mask = result_mask.reshape(img.shape)
    return result_mask, (str(sigma), str(alpha))

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters setting')
    parser.add_argument('path', type=str)
    parser.add_argument('--alpha', type=float, default=1e-2)
    parser.add_argument('--sigma', type=float, default=40)
    parser.add_argument('--seg_filename', default='tmp')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    file_dir = '../nrrd/cropped/npy/'
    result_dir = '../results/med_graphcut_res/'
    args = parse_args()

    img = np.load(file_dir + args.path)
    print(img.shape, img.max(), img.min())

    segmentation, ph = med_graphcut(img, args.alpha, args.sigma)
    segmentation = segmentation.astype(np.int)*255

    print('save the result')
    seg_filename = args.seg_filename #+'_%f_%f'%(args.alpha, args.sigma)
    np.save(result_dir + 'npy/' + seg_filename + '.npy', segmentation)
    nrrd.write(result_dir + seg_filename + '.nrrd', segmentation.astype(np.float))
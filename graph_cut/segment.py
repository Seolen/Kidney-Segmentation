import numpy as np
import scipy.misc as scm
import argparse
import matplotlib.pyplot as plt
import maxflow

import nrrd

file_dir = '../nrrd/cropped/'
result_dir = '../results/graph_cut_res/'

def create_graph(img, shape, k=1):
    # create graph
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(shape)
    structure = np.array([ [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                       [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                       [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
    g.add_grid_edges(nodeids, k,  structure=structure)
    g.add_grid_tedges(nodeids, img, 600 - img)
    # g.add_grid_tedges(nodeids[:, :, 0], np.inf, 0)
    # g.add_grid_tedges(nodeids[:, :, -1], 0, np.inf)

    return nodeids, g

def read_img(path):
    return scm.imread(path)

def show_seg(img2, img):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.imshow(img2[:, :, 0])
    plt.subplot(2, 2, 3)
    plt.imshow(img2[:, :, 1])
    plt.subplot(2, 2, 4)
    plt.imshow(img2[:, :, 2])
    plt.show()  # display

def save_npy_nrrd(npypath, nrrdpath, img2):
    print('save the result')
    np.save(result_dir+'npy/'+npypath, img2)
    nrrd.write(result_dir+npypath.split('.npy')[0]+'.nrrd', img2.astype(np.float))

if __name__ == '__main__':
    # setting
    # args = add_parsement()
    parser = argparse.ArgumentParser(description='Parameters setting')
    parser.add_argument('path', type=str)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--npy_path', default='tmp.npy')
    parser.add_argument('--nrrd_path', default='tmp.nrrd')
    args = parser.parse_args()

    img = np.load(file_dir + args.path)
    # img = img[200:400, 200:400, 200:400]
    img[img<0] = 0
    img[img>600] = 600
    print(img.shape, img.max(), img.min())

    nodeids, g = create_graph(img, img.shape, args.K)
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    print('grid segment got.')

    img2 = np.int_(np.logical_not(sgm))
    save_npy_nrrd(args.npy_path, args.nrrd_path, img2)
    # show_seg(img2, img)



    print('End')


# python segment.py P1_2.npy --K 0 --npy_path P1_2_K0.npy

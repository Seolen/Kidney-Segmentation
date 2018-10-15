import nrrd
import numpy as np
import cv2
from skimage.morphology import label
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion, generate_binary_structure
from multiprocessing import Pool
import time
import pickle
# from torch.multiprocessing import Pool
import argparse

def function(func, args):
    result = func(*args)
    return result

def get_size(start, end, mask):
    order_ind = list()
    for i in range(start, end):
        # print(i-start, "/", end-start)

        ### numpy version
        inds = np.where(mask==i)
        mask_size = inds[0].shape[0]

        ### pytorch version
        # inds = (new_mask == i).nonzero()
        # mask_size = inds.shape[0]
        if mask_size > 300:
            order_ind.append([i, mask_size])
    return order_ind


class Segmentation(object):
    def __init__(self, file, part_list=[1]):

        self.image = np.load(file)
        # self.image = nrrd.read(file)[0]

        # self.image, _ = nrrd.read(file)
        self.pool_num = 32
        print('image.shape: ', self.image.shape)

        self.part_list = part_list

    def get_mask(self):
        mask = binary_erosion(self.image)

        cca_mask, label_num = label(mask, neighbors=4, background=None, return_num=True, connectivity=None)
        # new_mask = torch.Tensor(new_mask).cuda()
        return cca_mask, label_num


    def get_tasks(self, piece_num, label_num, mask):
        TASKS = []
        for i in range(self.pool_num):
            start = i*piece_num
            end = (i+1)*piece_num if i!=self.pool_num-1 else label_num
            TASKS.append((get_size, (start, end, mask)))
        return TASKS


    def process(self, mid_dest):
        cca_mask, label_num = self.get_mask()
        piece_num = label_num//self.pool_num
        p = Pool(processes=self.pool_num)

        TASKS = self.get_tasks(piece_num, label_num, cca_mask)
        rets = [p.apply_async(function, t) for t in TASKS]

        start = time.time()
        index_list = []
        for r in rets:
            index_list.extend(r.get())
        end = time.time()
        print('time cost:', end - start)
        # import ipdb; ipdb.set_trace()

        with open(mid_dest, 'wb') as fp:
            ret = {}
            ret['mask'] = cca_mask
            ret['index'] = index_list
            pickle.dump(ret, fp)
        return cca_mask, label_num, index_list

    def segment(self, dest, mid_dest):
        cca_mask, label_num, index_list = self.process(mid_dest)
        all_mask = np.zeros(cca_mask.shape)
        if len(index_list)>1:
            sorted_ind = sorted(index_list, key=lambda x: x[1], reverse=True)

            # select ith largest
            # selected_inds = [sorted_ind[1], sorted_ind[2]]  # sorted_ind[2]
            selected_inds = []
            for ith in self.part_list:
                print(ith)
                selected_inds.append(sorted_ind[ith])

            for j in selected_inds:
                inds = np.where(cca_mask==j[0])
                print(inds[0].shape)
                all_mask[inds] = 1
        nrrd.write(dest, all_mask.astype(np.float))


if __name__ == '__main__':

    ori_dir = '../results/med_graphcut_res/'

    """
        find max connetction component
    """
    parser = argparse.ArgumentParser(description='source file, dest')
    parser.add_argument('file', help='The origin file path of the volume')
    parser.add_argument('phase', help='The phase of the volume')
    parser.add_argument('part', help='The part')
    parser.add_argument('--part_list', type=int, nargs='+', help='The ith part are chosen for final result, such as 2 3', required=True)
    # parser.add_argument('--k', type=int, help='The part')
    parser.add_argument('--regular', type=int, default=1, help='regular file path dir')
    args = parser.parse_args()

    alpha = 1e-3

    file_ori = args.file

    file = mid_dest = dest = None
    if args.regular == 1:
        file = ori_dir + 'npy/' + file_ori  # 1_artery_K0.npy
        mid_dest = ori_dir + 'seg/' + 'midres/' + '{}_{}'.format(args.phase, args.part)
        dest = ori_dir + 'seg/' + '{}_{}.nrrd'.format(args.phase, args.part)
    else:
        file = file_ori
        mid_dest = '../results/graph_cut_res/seg/midres_divided_essence_kidney'
        dest = '../results/graph_cut_res/seg/divided_essence_kidney_top2.nrrd'

    seg = Segmentation(file, args.part_list)
    seg.segment(dest, mid_dest)

    # python ... 1_artery_K0.npy arterial artery
    # python part_seg.py ../results/graph_cut_res/seg/divided_essence_kidney.nrrd asd asd --regular 0
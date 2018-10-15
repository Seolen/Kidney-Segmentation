import nrrd
import numpy as np
import cv2
from skimage.morphology import label
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion, generate_binary_structure
import torch
from multiprocessing import Pool
import time
import pickle


def function(func, args):
    result = func(*args)
    return result

def get_size(start, end, mask):
    order_ind = list()
    for i in range(start, end):
        print(i-start,"/", end-start)

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
    def __init__(self, file, part):
        self.image, _ = nrrd.read(file)
        self.pool_num = 32

        print(self.image.shape)
        self.kidney_upper = 284
        self.kidney_lower = 180#126
        self.artery_upper = 472
        self.artery_lower = 270
        self.vessel_upper = 472
        self.vessel_lower = 180
        self.tumor_upper = 180
        self.tumor_lower = 100

        if part == 'kidney':
            self.thresh_upper = self.kidney_upper
            self.thresh_lower = self.kidney_lower
        elif part == 'artery':
            self.thresh_upper = self.artery_upper
            self.thresh_lower = self.artery_lower
        elif part == 'vessel':
            self.thresh_upper = self.vessel_upper
            self.thresh_lower = self.vessel_lower
        elif part == 'tumor':
            self.thresh_upper = self.tumor_upper
            self.thresh_lower = self.tumor_lower

    def get_mask(self):
        mask = np.ones(self.image.shape)
        mask[np.where(self.image < self.thresh_lower)] = 0
        mask[np.where(self.image > self.thresh_upper)] = 0
        mask = binary_erosion(mask)

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

    def segment(self, dest, mid_dest, part):
        cca_mask, label_num, index_list = self.process(mid_dest)
        all_mask = np.zeros(cca_mask.shape)
        if len(index_list)>1:
            sorted_ind = sorted(index_list, key=lambda x: x[1], reverse=True)
            selected_inds = []

            if part == 'tumor':
                selected_inds = [sorted_ind[3]]
            elif part == 'kidney':
                selected_inds = [sorted_ind[1], sorted_ind[2]]
            elif part == 'artery':
                selected_inds = [sorted_ind[1]]
            elif part == 'vessel':
                selected_inds = [sorted_ind[1]]

            for j in selected_inds:
                inds = np.where(cca_mask==j[0])
                print(inds[0].shape)
                all_mask[inds] = 1
        nrrd.write(dest, all_mask.astype(np.float))

def get_filename(phase, part):
    file = mid_dest = dest = ''
    if phase == 'arterial':
        file = 'nrrd/1.nrrd'
        mid_dest = 'results/mid_res/mid_res_{}9'.format(part)
        dest = 'results/{}1.nrrd'.format(part)
    elif phase == 'essense':
        file = 'nrrd/2.nrrd'
        mid_dest = 'results/mid_res/mid_res_kidney6'
        dest = 'results/kidney6.nrrd'
    elif phase == 'vein':
        file = 'nrrd/3.nrrd'
        mid_dest = 'results/mid_res/mid_res_kidney_vein'
        dest = 'results/kidney_vein.nrrd'

    return file, mid_dest, dest

if __name__ == '__main__':
    """
    1.when phase = 'arterial' then file = 'nrrd/9.nrrd', if use:
        1.1. kidney thresh: related dest is kidney9_no_tumor.nrrd
        1.2. artery thresh: related dest is artery9.nrrd
        1.3. vessel_thresh: related dest is vessel9.nrrd
        1.4. tumor thresh: related dest is tumor9.nrrd
    2.when phase = 'essence', then file = 'nrrd/6.nrrd'
        must use kidney thresh to get kidney6.nrrd
    """
    phase = 'arterial' # (essence, arterial, vein)       (essence/kideney, vein/vessel, essense/tumor, arterial/)
    part = 'artery' #(kidney, vessel, artery, tumor)

    file = mid_dest = dest =None
    # if phase == 'arterial':
    #     file = 'nrrd/1.nrrd'
    #     mid_dest = 'results/mid_res/mid_res_{}9'.format(part)
    #     dest = 'results/{}1.nrrd'.format(part)
    # elif phase == 'essence':
    #     part = 'kidney'
    #     file = 'nrrd/2.nrrd'
    #     mid_dest = 'results/mid_res/mid_res_kidney6'
    #     dest = 'results/kidney6.nrrd'

    file, mid_dest, dest = get_filename(phase, part)

    seg = Segmentation(file, part)
    seg.segment(dest, mid_dest, part)
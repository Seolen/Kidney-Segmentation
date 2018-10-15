import nrrd
import numpy as np
import cv2
from skimage.morphology import label
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion, generate_binary_structure
from config import ARTERIAL_THRESH, VENOUS_THRESH, ESSENCE_THRESH
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
        print(i-start, "/", end-start)

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
    def __init__(self, file, part, threshold):
        self.image = np.load(file)
        # self.image, _ = nrrd.read(file)
        self.pool_num = 32

        print(self.image.shape)
        self.kidney_upper = threshold['kidney_upper']
        self.kidney_lower = threshold['kidney_lower']
        self.artery_upper = threshold['artery_upper']
        self.artery_lower = threshold['artery_lower']
        self.vein_upper = threshold['vein_upper']
        self.vein_lower = threshold['vein_lower']
        self.tumor_upper = threshold['tumor_upper']
        self.tumor_lower = threshold['tumor_lower']

        if part == 'kidney':
            self.thresh_upper = self.kidney_upper
            self.thresh_lower = self.kidney_lower
        elif part == 'artery':
            self.thresh_upper = self.artery_upper
            self.thresh_lower = self.artery_lower
        elif part == 'vein':
            self.thresh_upper = self.vein_upper
            self.thresh_lower = self.vein_lower
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
                selected_inds = [sorted_ind[1], sorted_ind[3]]
            elif part == 'artery':
                selected_inds = [sorted_ind[1]]
            elif part == 'vein':
                selected_inds = [sorted_ind[1]]

            for j in selected_inds:
                inds = np.where(cca_mask==j[0])
                print(inds[0].shape)
                all_mask[inds] = 1

        # fill holes
        all_mask = binary_fill_holes(all_mask)

        nrrd.write(dest, all_mask.astype(np.float))


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
    parser = argparse.ArgumentParser(description='Phase, part')
    parser.add_argument('file', help='The origin file path of the volume')
    parser.add_argument('phase', help='The phase of the volume')
    parser.add_argument('part', help='Choose a part to mask (artery, kidney and vein)')
    args = parser.parse_args()

    file_ori = args.file
    phase = args.phase
    part = args.part
    # phase = 'essence' # (essence, arterial, venous)
    # part = 'kidney' #(kidney, vein, artery, tumor)
    threshold = {}

    file = mid_dest = dest = None
    if phase == 'arterial':
        threshold = ARTERIAL_THRESH
        # file = 'nrrd/secondcase/1.npy'
        file = 'nrrd/secondcase/'+file_ori
        mid_dest = 'results/secondcase/mid_res/mid_res_arterial_{}'.format(part)
        dest = 'results/secondcase/{}_arterial_{}.nrrd'.format(file_ori.split('.')[0], part)
    elif phase == 'essence':
        threshold = ESSENCE_THRESH
        file = 'nrrd/secondcase/'+file_ori
        mid_dest = 'results/secondcase/mid_res/mid_res_essence_{}'
        dest = 'results/secondcase/{}_essence_{}.nrrd'.format(file_ori.split('.')[0], part)
    elif phase == 'venous':
        threshold = VENOUS_THRESH
        file = 'nrrd/secondcase/'+file_ori
        mid_dest = 'results/secondcase/mid_res/mid_res_venous_{}'
        dest = 'results/secondcase/{}_venous_{}.nrrd'.format(file_ori.split('.')[0], part)
    seg = Segmentation(file, part, threshold)
    seg.segment(dest, mid_dest, part)
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

def prob_map_md(img):
    '''
    It's a strong prior that higher pixel intensity corresponding to high probability to be foreground.
    Histogram is not suitable for this problem, So,
        probability = log(pixel_intensity)
    :return: Probability map. (foreground should be high, and background low)
    [prob values should be normalized]
    '''
    # the intensity may be 0, it is bad for calculation, so, shift 3 to right
    pmap = np.copy(img).astype(np.float)
    pmap = pmap + 1
    pmap *= 1 / pmap.max()
    pmap = (-1) * np.log(pmap)
    pmap *= 1/pmap.max()
    pmap = 1 - pmap

    return pmap



def prob_map(img, hist):
    '''
    :param img: (m,n) array or k-dimension array
    :param hist: a histogram model  (to do: what's the general representation?)
    :return: A probability map according to its pixel intensity,
             e.g. to get a foreground prob map, high intensity corresponds to high prob values.
    '''
    pmap = np.zeros_like(img).astype(np.float)
    # assume hist is a list, e.g. [0.25, 0.10, ...]
    k = hist.size
    flat_img = img.ravel()

    flag_array = np.zeros((k, flat_img.size))
    flag_array[flat_img, np.arange(flat_img.size)] = 1
    pmap = np.matmul(hist, flag_array).reshape(img.shape)

    # pmap: some value may be 0 or 1, be careful.
    print(pmap)
    import ipdb
    ipdb.set_trace()

    # -log(h(zn)), normalize, and reverse it by 1-x
    nl_map = (-1) * np.log(pmap)
    nl_map *= 1 / nl_map.max()
    nl_map = 1 - nl_map

    return nl_map

def prob_histogram(img, range, bins=256, density=True):
    '''
    Compute *Foreground* histogram. In practical, all pixels but background are involved in this hist.
    Note : bins value.
    :param img: img must be grayscale image array, and be in [0,255]. If not, PIL.Image.open().convert('L)
    :return:
    '''
    bin_density, bin_boundary = np.histogram(img, range=range, bins=range[1]-range[0]+1, density=density)
    complete = [0 for i in range(range[0])]
    bin_boundary = np.hstack([np.arange(range[0]), bin_boundary])
    bin_density = np.hstack([complete, bin_density])
    return bin_density



################
def test_prob_map(path='/home/seolen/seolen-tmp/wallpaper.jpg'):

    gray = False

    img = Image.open(path)
    if not gray:
        img = img.convert('L')
    img = np.array(img)

    histogram = prob_histogram(img)
    pmap = prob_map(img, prob_histogram(img))
    print(pmap.shape)
    print(pmap)
    print(np.sum(histogram))





if __name__ == '__main__':
    test_prob_map()
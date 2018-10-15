import numpy as np
import nrrd

base_dir = '../results/secondcase/'
seg_dir = '../results/graph_cut_res/seg/'

# reverse the file of artery+kidney
def reverse_npy(path=base_dir+'essence_artery_kidney.nrrd'):
    result_path = 'essence_artery_kidney2.npy'

    arr = nrrd.read(path)[0]
    new = np.zeros(arr.shape)
    slices = arr.shape[0]
    for i in range(slices):
        new[i] = arr[slices-1-i]
    np.save(base_dir+result_path, new)

    nrrd.write(path.split('.nrrd')[0]+'2.nrrd', new)

# divide artery and kidney
# artery(.nrrd) sum(.npy)
def divide_organ(f_sum, f_artery, f_mask):

    arr_artery = nrrd.read(f_artery)[0]
    arr_sum = np.load(f_sum)
    arr_mask = nrrd.read(f_mask)[0]

    arr_ext = np.zeros(arr_sum.shape)
    nonzero = np.nonzero(arr_mask)
    xmin, xmax, ymin, ymax, zmin, zmax = nonzero[0].min(), nonzero[0].max(), \
          nonzero[1].min(), nonzero[1].max(), nonzero[2].min(), nonzero[2].max()
    print(xmin, xmax, ymin, ymax, zmin, zmax)

    arr_ext[xmin:xmax, ymin:ymax, zmin:zmax] = arr_artery


    arr_kidney = np.subtract(arr_sum, arr_ext)
    arr_kidney[arr_kidney<0] = 0
    return arr_kidney

def save_nrrd(path, arr_kidney):
    nrrd.write(path, arr_kidney)

if __name__ == '__main__':

    # reverse_npy()
    file_sum = 'essence_artery_kidney2.npy'
    file_artery = 'essence_artery_k0.nrrd'
    file_mask = 'arterial_artery.nrrd'
    dest_path = 'divided_essence_kidney.nrrd'
    arr_kideny = divide_organ(base_dir+file_sum, seg_dir+file_artery, base_dir+file_mask)
    save_nrrd(seg_dir+dest_path, arr_kideny)
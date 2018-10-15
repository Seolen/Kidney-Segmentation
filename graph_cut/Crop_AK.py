import numpy as np
import nrrd

def origin_cropped(nrrdpath, npypath):
    bi_img = nrrd.read(nrrdpath)[0]
    ori_img = np.load(npypath)
    new_img = np.multiply(bi_img, ori_img)
    # save cropped image
    # nrrd.write(basedir+'1_artery_crop.nrrd', new_img)

    nonzero = np.nonzero(new_img)
    xmin, xmax, ymin, ymax, zmin, zmax = nonzero[0].min(), nonzero[0].max(), \
        nonzero[1].min(), nonzero[1].max(), nonzero[2].min(), nonzero[2].max()
    print(xmin, xmax, ymin, ymax, zmin, zmax)

    crop_img = ori_img[xmin:xmax, ymin:ymax, zmin:zmax]  # crop original part
    print(crop_img.shape)

    return crop_img

def Cropped_Box(nrrdpath):
    '''
    Given mask, get its box location, for latter comparation and processing.
    :param nrrdpath: The path of organ part mask, by the ways of thresholding.
    :return:
    '''
    bi_img = nrrd.read(nrrdpath)[0]
    nonzero = np.nonzero(bi_img)
    xmin, xmax, ymin, ymax, zmin, zmax = nonzero[0].min(), nonzero[0].max(), \
        nonzero[1].min(), nonzero[1].max(), nonzero[2].min(), nonzero[2].max()
    return (xmin, xmax, ymin, ymax, zmin, zmax)

def get_Boxs():

    nrrddir = '../nrrd/cropped/'
    nrrd_files = {'aa': '1_artery_artery.nrrd',
                  'ak': '1_artery_kidney.nrrd',
                  'ea': '1_essence_artery.nrrd',
                  'ek': '1_essence_kidney.nrrd'}
    for keys,values in nrrd_files.items():
        print(keys)
        box = Cropped_Box(nrrddir + values)
        print(box)
        print(box[1]-box[0], box[3]-box[2], box[5]-box[4])
        print('\n')


if __name__ == '__main__':
    get_Boxs()


    '''
    aa
    (1, 530, 209, 308, 182, 314)
    529 99 132
    
    
    ak
    (115, 360, 229, 357, 125, 418)
    245 128 293
    
    
    ek
    (1, 359, 197, 366, 127, 427)
    358 169 300
    
    
    Final:
    for artery: (1, 530, 209, 308, 182, 314)  [530, 100, 133]
    for kidney: (1, 360, 197, 366, 125, 427)  [361, 170, 203]
    
    '''



# basedir = '../nrrd/secondcase/'
# nrrdpath = '../results/secondcase/arterial_artery.nrrd'
# npypath = 'P1_2.npy'
# destpath = '../nrrd/cropped/1_essence_artery.npy'
#
# img2 = origin_cropped(nrrdpath, basedir + npypath)
# np.save(destpath, img2)
# nrrd.write(destpath.split('npy')[0]+'nrrd', img2)


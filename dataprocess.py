import os
import pydicom
import nrrd
import numpy as np
df = './patients.txt'
# datadir = './CTav/DICOM/S31040/S5010/'
# files = os.listdir(datadir)

patients = [] # the patients
files = [] # the series
tmp = []   # imgs in one series
cube = []


with open(df) as f:
    f.readline()
    for line in f:
        address = line.strip().replace('/home/seolen/disk/', '')
        if address == '###':
            files.append(tmp)
            tmp = []
            f.readline()
        else:
            tmp.append(address)

# for ind, file in enumerate(files[1:-1]):
#     cube = []
#     for i in file:
#         img = pydicom.dcmread(i).pixel_array
#         cube.append(img)
#     cube = np.array(cube).astype(np.int32)-1000
#     nrrd.write('./nrrd/secondcase/{}.nrrd'.format(str(ind+2)), cube)
for i in files[3]:
    img = pydicom.dcmread(i).pixel_array
    cube.append(img)
cube = np.array(cube).astype(np.int32)-1000
nrrd.write('./nrrd/secondcase/4.nrrd', cube)
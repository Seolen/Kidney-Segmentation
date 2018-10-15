#!/bin/sh
file="1_essence_artery.npy"
alpha = 0
sigma = 0
phase="essence"
part="artery"
dest="$phase_$part"
python segment.py $file --npy_path $dest
python part_seg.py "$phase_$part.npy" $phase $part

# python med_graphcut.py 1_artery_kidney.npy --seg_filename a_kidney
# python part_seg.py a_kidney.npy a kidney --part_list 2 3
# python part_seg.py a_artery.npy a artery --part_list 1
# Process
1. Run kidney_3dseg.py for 5 times to get kidney9.nrrd/artery9.nrrd/vessel9.nrrd/tumor9.nrrd/kidney6.nrrd (note that kidney9 contain part of vein, while kidney6 is clean but have some rotation and transfomation from kidney9)
2. Run postprocess.ipynb to:
    2.1 Flood Fill for vessel/kidney/artery
    2.2 Get tiny vessel by kidney9 and artery9 ablation in vessel9 model
4. Run get_vein.ipynb to separate vein and kidney in kidney9 by ablation operation with kidney6



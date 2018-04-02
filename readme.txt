

Requirements: software
GPU
caffe
senet:https://github.com/hujie-frank/SENet
opencv3.2
python2.7
matplotlib
skimage

rebuild senet caffe

change the IPIU_IGRSS_2018 in run.sh to your own path
cd IPIU_IGRSS_2018
sh run.sh


Preparation for Training

    change the lidar_dir path in data/lidar_analysis.py to your own path
    run python data/lidar_analysis.py to creat the lidar data for train and test.

    using the ENVI convert the original HSI data into TIFF, and change the sbdd_dir path in data/hsi_analysis.py to the path for TIFF HSI
    run python data/hsi_analysis.py to creat the hsi data for train and test.

    change the sbdd_dir2_path  in data/vhr_analysis.py to your own path
    run python data/vhr_analysis.py to creat the vhr data for road detection.

    change the gt_path in gt_16.py to your own gt path
    run python gt_16.py convert ground truth to 16 category and 5 category

    run python model/train_idx_fusion.py to creat the train coordinate in training data for fusion network

Train and Test Fusion Network

    cahnge the caffe path in solve_fusion.py、solve_senet.py、voc_layers_fusion.py、voc_layers_fusion.py、test_fusion.py、test_senet.py
    run python model/solve_fusion.py to train the network
    run python model/test_fusion.py to test the network

    run python road_detection/all_canny.py to detection road

    run python model/train_idx_senet.py to creat the train coordinate in training data for road senet network

    run python model/solve_senet.py to train the road network
    run python model/test_senet.py to test the road network

    run python result_combine.py to combine the final result

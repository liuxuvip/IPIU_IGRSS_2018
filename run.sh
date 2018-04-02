
cd IPIU_IGRSS_2018

cd data

### data denosing and normalization
python lidar_analysis.py
python hsi_analysis.py
python vhr_analysis.py


cd ..

###  convert ground truth to 16 category and 5 category
python gt_16.py


cd model
### creat the train coordinate in training data for fusion network
python train_idx_fusion.py

### train and test the fusion network
python solve_fusion.py
python test_fusion.py

cd ..
cd road_detection
### road detection
python all_canny.py

cd ..
cd model
### creat the train coordinate in training data for senet
python train_idx_senet.py

### train and test the senet
python solve_senet.py
python test_senet.py

cd ..
### result combine
python result_combine.py

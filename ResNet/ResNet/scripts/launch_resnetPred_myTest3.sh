
## to train:

# for resnet34:
python resnetPred_myTest3.py --batch_size 1 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/ \
 --label_name DSM_Anxi_T --tolerance 0.2
 
# for resnet9:
python resnetPred_myTest3.py --batch_size 2 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/ \
 --label_name DSM_Anxi_T --tolerance 0.2


# to test:

# for resnet34:
python testModel_acc_v2.py --model_path results_myTest3/best_1_50_model.pth \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/ \
 --label_name DSM_Anxi_T --batch_size 1 \
 --result_dir results_myTest3/acc_v2.txt --tolerance 0.2

# for resnet9:
python testModel_acc_v2.py --model_path results_myTest3/best_2_50_model.pth \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/ \
 --label_name DSM_Anxi_T --batch_size 2 \
 --result_dir results_myTest3/acc_v2.txt --tolerance 0.2




  
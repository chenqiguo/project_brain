
## to train:

# NOT USED:
#python resnetPred_myTest2.py --batch_size 1 --epochs 50 \
# --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/ \
# --label_name DSM_Anxi_T --train_split 0.7 --tolerance 2

python myModelArchPred_train.py --batch_size 1 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/ \
 --label_name DSM_Anxi_T --train_split 0.7 --tolerance 0.2 \
 --result_rootDir /eecf/cbcsl/data100b/Chenqi/project_brain/results_customModel/ \
 --input_height 64984 --input_width 379 --model_type colKernel

python myModelArchPred_train.py --batch_size 10 --epochs 50 \
 --label_name DSM_Anxi_T --tolerance 0.2 \
 --input_height 64984 --input_width 379 --model_type rowKernel --model_version v4

python myModelArchPred_train.py --batch_size 16 --epochs 50 \
 --label_name DSM_Anxi_T --tolerance 0.2 \
 --input_height 64984 --input_width 379 --model_type rowcolKernel --model_version v4


# to test: NOT did !!!
python testModel_acc_v1.py --model_path results_myTest2/best_1_50_model.pth \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/ \
 --label_name DSM_Anxi_T --batch_size 1 \
 --result_dir results_myTest2/acc_v1.txt --tolerance 2

  
  
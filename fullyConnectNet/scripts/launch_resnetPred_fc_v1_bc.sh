
## to train:

# for fc v1 2PCA_v3 flatten all subjects binary classification (cross_entropy loss) DSM_Anxi_T:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --label_name DSM_Anxi_T --result_dir results/fc_v1_bc_allSubjects_thresh50/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/

# for fc v1 2PCA_v4 flatten all subjects binary classification (cross_entropy loss) DSM_Anxi_T:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4/connData_dimReduc_PCA_dict_v4.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4/maxlr_0_05/fc_v1_bc_allSubjects_thresh50/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/



# to test:

# for fc v1 2PCA_v3 flatten all subjects binary classification DSM_Anxi_T:
python testModel_fc_v1_bc.py --model_path results/fc_v1_bc_allSubjects_thresh50/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/fc_v1_bc_allSubjects_thresh50/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/

# for fc v1 2PCA_v4 flatten all subjects binary classification (cross_entropy loss) DSM_Anxi_T:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4/maxlr_0_05/fc_v1_bc_allSubjects_thresh50/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4/maxlr_0_05/fc_v1_bc_allSubjects_thresh50/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4/connData_dimReduc_PCA_dict_v4.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/

 


  
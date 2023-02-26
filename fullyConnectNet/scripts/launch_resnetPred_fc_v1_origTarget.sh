
## to train:

# for fc v1 2PCA flatten all subjects (L1 loss):
python resnetPred_fc_v1_origTarget.py --batch_size 100 --epochs 150 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --label_name DSM_Anxi_T --tolerance 2 --result_dir results/fc_v1_origTarget_allSubjects/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/

# for fc v1 2PCA flatten all subjects (PearsonCorr v1 loss):
python resnetPred_fc_v1_origTarget_PearsonCorr.py --batch_size 100 --epochs 150 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --label_name DSM_Anxi_T --result_dir results/fc_v1_origTarget_allSubjects_lossPearsonCorr_v1/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/

# for fc v1 2PCA flatten ONLY target>50 subjects (L1 loss):
python resnetPred_fc_v1_origTarget.py --batch_size 100 --epochs 150 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --label_name DSM_Anxi_T --tolerance 2 --result_dir results/fc_v1_origTarget_greater50Subjects/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/fullyConnectNet/train_test_split/only_targetGreater50/

# for fc v1 2PCA flatten ONLY target>50 subjects (PearsonCorr v1 loss):
python resnetPred_fc_v1_origTarget_PearsonCorr.py --batch_size 100 --epochs 150 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --label_name DSM_Anxi_T --result_dir results/fc_v1_origTarget_greater50Subjects_lossPearsonCorr_v1/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/fullyConnectNet/train_test_split/only_targetGreater50/


# for fc v1 2PCA_v4 flatten all subjects regression (l1 loss) DSM_Anxi_T:
python resnetPred_fc_v1_origTarget.py --batch_size 100 --epochs 150 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4/connData_dimReduc_PCA_dict_v4.pkl \
 --label_name DSM_Anxi_T --tolerance 2 --result_dir results/DSM_Anxi_T/PCA_v4/fc_v1_origTarget_allSubjects/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/






# to test:

# for fc v1 2PCA flatten all subjects (L1 loss):
python testModel_fc_v1_origTarget.py --model_path results/fc_v1_origTarget_allSubjects/model_weights_bestValidLoss.pth \
 --batch_size 100 --label_name DSM_Anxi_T --tolerance 2 --result_dir results/fc_v1_origTarget_allSubjects/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/

# for fc v1 2PCA flatten all subjects (PearsonCorr v1 loss):
python testModel_fc_v1_origTarget_PearsonCorr.py --model_path results/fc_v1_origTarget_allSubjects_lossPearsonCorr_v1/model_weights_bestValidLoss.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/fc_v1_origTarget_allSubjects_lossPearsonCorr_v1/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/

# for fc v1 2PCA flatten ONLY target>50 subjects (L1 loss):
python testModel_fc_v1_origTarget.py --model_path results/fc_v1_origTarget_greater50Subjects/model_weights_bestValidLoss.pth \
 --batch_size 100 --label_name DSM_Anxi_T --tolerance 2 --result_dir results/fc_v1_origTarget_greater50Subjects/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/fullyConnectNet/train_test_split/only_targetGreater50/

# for fc v1 2PCA flatten ONLY target>50 subjects (PearsonCorr v1 loss):
python testModel_fc_v1_origTarget_PearsonCorr.py --model_path results/fc_v1_origTarget_greater50Subjects_lossPearsonCorr_v1/model_weights_bestValidLoss.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/fc_v1_origTarget_greater50Subjects_lossPearsonCorr_v1/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/fullyConnectNet/train_test_split/only_targetGreater50/


# for fc v1 2PCA_v4 flatten all subjects regression (l1 loss) DSM_Anxi_T:
python testModel_fc_v1_origTarget.py --model_path results/DSM_Anxi_T/PCA_v4/fc_v1_origTarget_allSubjects/model_weights_bestValidLoss.pth \
 --batch_size 100 --label_name DSM_Anxi_T --tolerance 2 --result_dir results/DSM_Anxi_T/PCA_v4/fc_v1_origTarget_allSubjects/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4/connData_dimReduc_PCA_dict_v4.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/


  
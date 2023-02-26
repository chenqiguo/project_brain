
## to train with 10-fold Cross Validation:

# for fc v1 2PCA_v4 flatten all subjects binary classification (cross_entropy loss) DSM_Anxi_T:
# CV0:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV0/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV0/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV0/
# CV1:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV1/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV1/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV1/
# CV2:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV2/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV2/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV2/
# CV3:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV3/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV3/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV3/
# CV4:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV4/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV4/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV4/
# CV5:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV5/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV5/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV5/
# CV6:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV6/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV6/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV6/
# CV7:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV7/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV7/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV7/
# CV8:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV8/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV8/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV8/
# CV9:
python resnetPred_fc_v1_bc.py --batch_size 100 --epochs 50 \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV9/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV9/ \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV9/



# to test with 10-fold Cross Validation:

# for fc v1 2PCA_v4 flatten all subjects binary classification (cross_entropy loss) DSM_Anxi_T:
# CV0:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV0/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV0/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV0/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV0/
# CV1:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV1/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV1/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV1/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV1/
# CV2:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV2/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV2/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV2/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV2/
# CV3:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV3/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV3/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV3/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV3/
# CV4:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV4/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV4/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV4/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV4/
# CV5:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV5/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV5/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV5/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV5/ 
# CV6:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV6/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV6/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV6/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV6/ 
# CV7:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV7/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV7/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV7/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV7/ 
# CV8:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV8/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV8/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV8/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV8/ 
# CV9:
python testModel_fc_v1_bc.py --model_path results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV9/model_weights_bestValidAcc.pth \
 --batch_size 100 --label_name DSM_Anxi_T --result_dir results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/CV9/ \
 --data_dir /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/CV9/connData_dimReduc_PCA_dict_v4_CV.pkl \
 --train_test_root /eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/CV9/ 





  
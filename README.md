# DAD-SGM
An implementation of the IEEE Transactions on Artificial Intelligence "Diffusion-Assisted Distillation for Self-Supervised Graph Representation Learning with MLPs" (Under Minor Revision).  
Thank you for your interest in our work!  

# Motivation
We find out that current GNN-to-MLP distillation methods often fail to preserve the task-agnostic knowledge learned by self-supervised GNN teachers.  
Hence, we introduce a new research direction for distilling task-agnostic knowledge from self-supervised graph neural networks (GNNs) to multi-layer perceptrons (MLPs).  
This approach enhances the noise robustness of MLPs against input noise, thereby producing robust node representations.  

![image](https://github.com/user-attachments/assets/62a9d532-e3ab-4bca-8a79-a8c5b59495de)


# Setup
Python 3.9.18   
Pytorch 1.9.0  
torch_geometric 2.0.4  
numpy 1.26.3  

# Teacher Model
We provide several pre-trained outputs of teacher models in the 'teacher_model' folder.  
You can access the teacher models' implementations as follows:  
DGI [https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_inductive.py]  
GRACE [https://github.com/CRIPAC-DIG/GRACE]  
CCA-SSG [https://github.com/hengruizhang98/CCA-SSG]  

# Run
python model/main.py

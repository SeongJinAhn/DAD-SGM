# DAD-SGM
(Revision under IEEE Transactions on Artificial Intelligence)  
We introduce a new research direction for distilling task-agnostic knowledge from self-supervised GNNs to MLPs.  
To achieve this, we design a GNN-to-MLP distillation that employs a diffusion teacher-assistant model.  
This approach drastically enhances the noise robustness of MLPa against input noise, thereby producing robust node representations with MLPs.  

![image](https://github.com/user-attachments/assets/62a9d532-e3ab-4bca-8a79-a8c5b59495de)


# Setup
Python 3.9.18   
Pytorch 1.9.0  
torch_geometric 2.0.4  
numpy 1.26.3  

# Teacher Model
We provide several pre-trained outputs of teacher models in the 'teacher_model' folder.
DGI [https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_inductive.py]
GRACE [https://github.com/CRIPAC-DIG/GRACE]
CCA-SSG [https://github.com/hengruizhang98/CCA-SSG]

# Run
python model/main.py

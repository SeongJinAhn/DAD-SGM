import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, load_loss_fn, load_simple_loss_fn
from utils.logger import set_log
from utils.metric import svm_test2, LogReg
import pickle

class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.x, self.y, self.adj, self.train_mask, self.valid_mask, self.test_mask = load_data(self.config)
        self.losses = load_loss_fn(self.config, self.device)
        
    def train(self, ts, teacher):
        self.teacher = teacher
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # Prepare model, optimizer, and logger
        self.params = load_model_params(self.config)
        self.model, self.optimizer, _ = load_model_optimizer(self.params, self.config.train, self.device)
        self.loss_fn = self.losses.loss_fn
        self.estimator = self.losses.estimate
        label = torch.argmax(self.y, 1)

        self.adj = None
        file_name = './teacher_model/' + self.teacher + '_' + self.config.data.data + '.txt'       # Outputs of teacher models
        with open(file_name, 'rb') as f:
           teacher_h, self.adj, self.x = pickle.load(f)[:3]
           teacher_h = teacher_h.cuda()
           teacher_h = (teacher_h - teacher_h.mean(0)) / (teacher_h.std(0)+1e-6)
           teacher_h = teacher_h.detach()
           self.adj = self.adj.cuda()
           self.x = self.x.cuda()

        file_name = './position/'+self.config.data.data+'_pe.txt'
        with open(file_name, 'rb') as f:
            self.pe = pickle.load(f).detach()
            self.x = torch.cat([self.x, self.pe], 1)

        print('')
        print('Stage 1: Training TA Model')
        text = './teacher_assistant/diffusion_' + self.teacher.lower() +'_'+self.config.data.data+'.pt'

        best_score  = 0
        # Diffusion Distillation Model
        for epoch in range(0, self.config.train.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            dropout_x = F.dropout(self.x, p=0.5)
            loss_subject = (dropout_x, None,  teacher_h, label, self.train_mask, self.config.train.time_batch)
            loss = self.loss_fn(self.model, *loss_subject)
            loss.backward()
            self.optimizer.step()
            # Evaluate the model
            if epoch % self.config.train.print_interval == 0:
                with torch.no_grad():
                    y_est = self.estimator(self.model, self.x, self.adj, teacher_h, label, 0)
                    y_est = F.normalize(y_est, p=2, dim=1)
                    print('Epoch :', epoch, ', SVM :', svm_test2(y_est.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy())[1], ', LogReg :', LogReg(y_est.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy())) #LogReg(y_est.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy()))

                    if LogReg(y_est.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy()) > best_score:
                        best_score = LogReg(y_est.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy())
                        torch.save(self.model.state_dict(), text)

        self.model.load_state_dict(torch.load(text))
        self.model.eval()

        print('----------------------------------------')
        print('Stage 2: Training MLP Student')
        # Diffusion Distillation Model
        best_student = 0
        self.student_model, self.student_optimizer, _ = load_model_optimizer(self.params, self.config.train, self.device)
        for epoch in range(0, 501):
            self.student_model.train()
            self.student_optimizer.zero_grad()
            student_h = self.student_model.student_forward(self.x, train=True)
            loss_subject = (self.x, self.adj, student_h, teacher_h, label, self.train_mask, self.config.train.time_batch)
            loss = self.losses.sds(self.model, *loss_subject)
            loss.backward()
            self.student_optimizer.step()
                
            # Evaluate the model
            if epoch % 5 == 0:
                with torch.no_grad():
                    student_h = self.student_model.student_forward(self.x, train=False)
                    student_h = F.normalize(student_h, p=2, dim=1)
                    print('Epoch :', epoch, ', SVM :', svm_test2(student_h.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy())[1], ', LogReg :', LogReg(student_h.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy())) #LogReg(y_est.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy()))

                    if best_student < LogReg(student_h.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy()):
                        best_student = LogReg(student_h.cpu().detach().numpy(), label.cpu().detach().numpy(), self.train_mask.cpu().detach().numpy(), self.test_mask.cpu().detach().numpy())

        print(best_student)
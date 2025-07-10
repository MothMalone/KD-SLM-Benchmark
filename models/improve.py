import torch
import torch.functional.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 排序不正确，最优传输对应关系不对
def improved_sort(value):
    sums = value.sum(dim=-1)
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_student = value[sorted_indices]
    sorted_values = sorted_student.values
    return sorted_values

# 模型能力不同，方差不一致
def normalize(value,T=2):
    means = value.mean(dim=-1, keepdim=True)
    stds = value.std(dim=-1, keepdim=True)
    z_score_normalized_student = (value - means) / (stds+0.0001)
    return z_score_normalized_student

# 有的点不重要
def trunc(s,t,d=100):
    return(s[:,:,:d],t[:,:,:d])

# 少量点用KL
def KL_wo(y_s, y_t,T=2):
    p_s = F.log_softmax(y_s, dim=-1)
    p_t = F.softmax(y_t, dim=-1)
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss

def KL_w(y_s, y_t,T=2):
    p_t = normalize(y_t,T)
    p_s = math.log(normalize(y_s,T))
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss

# seq level 最优传输
class Sinkhorn_seq(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = 2   
    def sinkhorn_normalized(self,x, n_iters=10):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self,x, y, epsilon=0.1, n_iters=20):
        Wxy = torch.cdist(x, y, p=1)  
        K = torch.exp(-Wxy / epsilon)  
        P = self.sinkhorn_normalized(K, n_iters)  
        return torch.sum(P * Wxy)  # 计算近似 EMD 损失
    def forward(self, y_s, y_t):
        softmax = nn.Softmax(dim=1)
        p_s = softmax(y_s/self.T)
        p_t = softmax(y_t/self.T)
        emd_loss = 0
        for i in range(p_s.shape[0]):
            emd_loss = 0.001*self.sinkhorn_loss(x=p_s[i],y=p_t[i])
        return emd_loss
    
class Sinkhorn_seq_w(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq_w, self).__init__()
        self.T = 2   
    def sinkhorn_normalized(self,x, n_iters=10):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self,x, y, epsilon=0.1, n_iters=20):
        Wxy = torch.cdist(x, y, p=1)  # 计算成本矩阵
        K = torch.exp(-Wxy / epsilon)  # 计算内核矩阵
        P = self.sinkhorn_normalized(K, n_iters)  # 计算 Sinkhorn 迭代的结果
        return torch.sum(P * Wxy)  # 计算近似 EMD 损失
    def forward(self, y_s, y_t):
        emd_loss = 0
        p_s = normalize(y_s,2)
        p_t = normalize(y_t,2)
        for i in range(p_s.shape[0]):
            emd_loss = 0.001*self.sinkhorn_loss(x=p_s[i],y=p_t[i])
        return emd_loss

# word level 最优传输   
class Sinkhorn_word(nn.Module):
    def __init__(self, epsilon=0.1, max_iter=20, reduction='mean',T=2):
        super(Sinkhorn_word, self).__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.reduction = reduction
        self.T = T

    def forward(self, teacher_logits, student_logits):
        # Ensure logits are normalized to form probability distributions
        teacher_probs = F.softmax(teacher_logits/self.T, dim=-1)
        student_probs = F.softmax(student_logits/self.T, dim=-1)

        batch_size, seq_length, num_classes = teacher_probs.size()

        # Initialize loss
        total_loss = 0.0

        for i in range(seq_length):
            teacher_probs_i = teacher_probs[:, i, :]
            student_probs_i = student_probs[:, i, :]

            # Compute cost matrix (0-1 matrix)
            cost_matrix = self._cost_matrix(num_classes, teacher_probs.device)
        
            # Apply Sinkhorn algorithm
            transport_matrix = self._sinkhorn(teacher_probs_i, student_probs_i, cost_matrix)

            # Compute Sinkhorn loss for this token
            loss = torch.sum(transport_matrix * cost_matrix)
            total_loss += loss

        # Average loss over sequence length
        total_loss /= seq_length

        if self.reduction == 'mean':
            total_loss = total_loss / batch_size

        return total_loss

    def _cost_matrix(self, num_classes, device):
        return 1 - torch.eye(num_classes, device=device)

    def _sinkhorn(self, teacher_probs, student_probs, cost_matrix):
        batch_size, num_classes = teacher_probs.size(0), teacher_probs.size(1)
        
        u = torch.zeros(batch_size, num_classes, device=teacher_probs.device)
        v = torch.zeros(batch_size, num_classes, device=student_probs.device)

        K = torch.exp(-cost_matrix / self.epsilon)
        K_tilde = K / torch.sum(K, dim=-1, keepdim=True)

        for _ in range(self.max_iter):
            u = 1.0 / (torch.matmul(K_tilde, student_probs / torch.matmul(K, u.unsqueeze(-1))).squeeze(-1))
            v = 1.0 / (torch.matmul(K_tilde.transpose(-1, -2), teacher_probs / torch.matmul(K.transpose(-1, -2), v.unsqueeze(-1))).squeeze(-1))

        transport_matrix = torch.matmul(torch.diag_embed(u), torch.matmul(K, torch.diag_embed(v)))
        return transport_matrix
    
class Sinkhorn_word_w(nn.Module):
    def __init__(self, epsilon=0.1, max_iter=20, reduction='mean',T=2):
        super(Sinkhorn_word_w, self).__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.reduction = reduction
        self.T = T

    def forward(self, teacher_logits, student_logits):
        # Ensure logits are normalized to form probability distributions
        teacher_probs = normalize(teacher_logits,self.T)
        student_probs = normalize(student_logits,self.T)

        batch_size, seq_length, num_classes = teacher_probs.size()

        # Initialize loss
        total_loss = 0.0

        for i in range(seq_length):
            teacher_probs_i = teacher_probs[:, i, :]
            student_probs_i = student_probs[:, i, :]

            # Compute cost matrix (0-1 matrix)
            cost_matrix = self._cost_matrix(num_classes, teacher_probs.device)
        
            # Apply Sinkhorn algorithm
            transport_matrix = self._sinkhorn(teacher_probs_i, student_probs_i, cost_matrix)

            # Compute Sinkhorn loss for this token
            loss = torch.sum(transport_matrix * cost_matrix)
            total_loss += loss

        # Average loss over sequence length
        total_loss /= seq_length

        if self.reduction == 'mean':
            total_loss = total_loss / batch_size

        return total_loss

    def _cost_matrix(self, num_classes, device):
        return 1 - torch.eye(num_classes, device=device)

    def _sinkhorn(self, teacher_probs, student_probs, cost_matrix):
        batch_size, num_classes = teacher_probs.size(0), teacher_probs.size(1)
        
        u = torch.zeros(batch_size, num_classes, device=teacher_probs.device)
        v = torch.zeros(batch_size, num_classes, device=student_probs.device)

        K = torch.exp(-cost_matrix / self.epsilon)
        K_tilde = K / torch.sum(K, dim=-1, keepdim=True)

        for _ in range(self.max_iter):
            u = 1.0 / (torch.matmul(K_tilde, student_probs / torch.matmul(K, u.unsqueeze(-1))).squeeze(-1))
            v = 1.0 / (torch.matmul(K_tilde.transpose(-1, -2), teacher_probs / torch.matmul(K.transpose(-1, -2), v.unsqueeze(-1))).squeeze(-1))

        transport_matrix = torch.matmul(torch.diag_embed(u), torch.matmul(K, torch.diag_embed(v)))
        return transport_matrix
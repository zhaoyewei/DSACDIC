import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss

class LMMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st, weight_ts = self.cal_weight(s_label, t_label)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()
        weight_ts = torch.from_numpy(weight_ts).cuda()

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        TS = kernels[batch_size:,:batch_size]

        loss += torch.sum(weight_ss * SS) + torch.sum(weight_tt * TT) - torch.sum(weight_st * ST) - torch.sum(weight_ts * TS)
        return loss


    def cal_weight(self, s_label, t_label):
        source_batch_size = s_label.size()[0]
        target_batch_size = t_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = np.eye(self.class_num)[s_sca_label]
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, self.class_num)
        s_sum[s_sum == 0] = 1e9
        s_vec_label = s_vec_label / s_sum


        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        # t_vec_label = np.eye(self.class_num)[t_sca_label]
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, self.class_num)
        t_sum[t_sum == 0] = 1
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        s_mask_arr = np.zeros((source_batch_size, self.class_num))
        s_mask_arr[:, index] = 1
        s_vec_label = s_vec_label * s_mask_arr

        t_mask_arr = np.zeros((target_batch_size, self.class_num))
        t_mask_arr[:, index] = 1
        t_vec_label = t_vec_label * t_mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)
        weight_ts = np.matmul(t_vec_label, s_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
            weight_ts = weight_ts / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
            weight_ts = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32'), weight_ts.astype("float32")

class DMMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(DMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label.cpu().data.numpy()
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        indices = list(set(s_sca_label) & set(t_sca_label))
        for index in indices:
            si_batch_size = (s_sca_label==index).sum().item()
            si = source[s_sca_label==index]

            # ti_batch_size = (t_sca_label==index).sum().item()
            ti = target
            tw = t_label.cpu().data[:,index].numpy()
            ti_batch_size = ti.shape[0]

            kernels = self.guassian_kernel(si, ti, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            #(B1,B1),(B2,B2),(B1,B2),(B2,B1)
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]

            tw_tt = torch.from_numpy(tw.reshape(-1,1) @ tw.reshape(1,-1)).cuda()
            tw_st = torch.from_numpy(tw.reshape(1,-1)).cuda()
            tw_ts = torch.from_numpy(tw.reshape(-1,1)).cuda()
            # loss += (torch.sum(SS) + torch.sum(tw_tt * TT) - torch.sum(ST * tw_st) - torch.sum(tw_ts * TS))/(si_batch_size+ti_batch_size)
            loss += torch.sum(SS)/(si_batch_size*si_batch_size) + torch.sum(tw_tt * TT)/(ti_batch_size*ti_batch_size) - torch.sum(ST * tw_st)/(si_batch_size*ti_batch_size) - torch.sum(tw_ts * TS)/(ti_batch_size*si_batch_size)
        loss/=len(indices)
        return loss

class class_MMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(class_MMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label.cpu().data.numpy()
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        indices = list(set(s_sca_label) & set(t_sca_label))
        for index in indices:
            si_batch_size = (s_sca_label==index).sum().item()
            ti_batch_size = (t_sca_label==index).sum().item()
            si = source[s_sca_label==index]
            ti = target[t_sca_label==index]
            kernels = self.guassian_kernel(si, ti,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            kernels /= (si_batch_size+ti_batch_size) ** 2
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            # loss += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))/(si_batch_size+ti_batch_size)
            loss += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))

        loss/=len(indices)
        return loss
#9专用
class weighted_class_MMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(weighted_class_MMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label.cpu().data.numpy()
        t_sca_prob = t_label.cpu().data.max(1)[0].numpy()
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        indices = list(set(s_sca_label) & set(t_sca_label))
        for index in indices:
            si_batch_size = (s_sca_label==index).sum().item()
            ti_batch_size = (t_sca_label==index).sum().item()
            si = source[s_sca_label==index]
            # si /= si.sum(axis=1,keepdims=True)
            si = F.normalize(si,p=2,dim=1)
            ti = target[t_sca_label==index]
            # ti /= ti.sum(axis=1,keepdims=True)
            ti = F.normalize(ti,p=2,dim=1)
            #(B2)
            tw = t_sca_prob[t_sca_label==index]
            kernels = self.guassian_kernel(si, ti,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            #(B1,B1),(B2,B2),(B1,B2),(B2,B1)
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            tw_tt = torch.from_numpy(tw.reshape(-1,1) @ tw.reshape(1,-1)).cuda()
            tw_st = torch.from_numpy(tw.reshape(1,-1)).cuda()
            tw_ts = torch.from_numpy(tw.reshape(-1,1)).cuda()
            # loss += (torch.sum(SS) + torch.sum(tw_tt * TT) - torch.sum(ST * tw_st) - torch.sum(tw_ts * TS))/(si_batch_size+ti_batch_size)
            loss += torch.sum(SS)/(si_batch_size*si_batch_size) + torch.sum(tw_tt * TT)/(ti_batch_size*ti_batch_size) - torch.sum(ST * tw_st)/(si_batch_size*ti_batch_size) - torch.sum(tw_ts * TS)/(ti_batch_size*si_batch_size)
        loss/=len(indices)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32') 

class st_class_MMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(st_class_MMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def get_loss(self, source, target, s_label, t_label,num_source_data):
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label[num_source_data:].cpu().data.numpy()
        source = source[num_source_data:]
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        indices = list(set(s_sca_label) & set(t_sca_label))
        for index in indices:
            si_batch_size = (s_sca_label==index).sum().item()
            ti_batch_size = (t_sca_label==index).sum().item()
            si = source[s_sca_label==index]
            ti = target[t_sca_label==index]
            kernels = self.guassian_kernel(si, ti,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            kernels /= (si_batch_size+ti_batch_size)**2
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            #2023-03-06把这个地方的数字修改了一下
            loss += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))
        loss/=len(indices)
        return loss

class stMMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,temperature=0.07,base_temperature=0.07):
        super(stMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.temperature = temperature
        self.base_temperature = base_temperature

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def source_guassian_kernel(self, source, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])
        total = source
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    
    def get_contrast_loss(self,source,s_label):
        batch_size = source.shape[0]
        kernels = self.source_guassian_kernel(source, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        if torch.sum(torch.isnan(sum(kernels))): return torch.tensor([0.]).cuda()
        s_label = s_label.contiguous().view(-1, 1)
        mask = torch.eq(s_label, s_label.T).float().cuda()
        # compute logits
        # anchor_dot_contrast = kernels
        anchor_dot_contrast = torch.div(kernels,self.temperature*self.kernel_num)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        non_zero_mask = mask.sum(1)!=0
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask[non_zero_mask] * log_prob[non_zero_mask]).sum(1) / mask[non_zero_mask].sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def source_guassian_kernel2(self, source, kernel_mul=2.0, kernel_num=5, fix_sigma=None,temperature=1.):
        n_samples = int(source.size()[0])
        total = source
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernels =  [-L2_distance / bandwidth_temp for bandwidth_temp in bandwidth_list]
        # kernel_vals = [-L2_distance / bandwidth_temp / temperature for bandwidth_temp in bandwidth_list]
        # kernel_val = [_kernel_val - torch.max(_kernel_val,dim=1,keepdim=True)[0].detach() for _kernel_val in kernel_vals]
        # kernel_val = [torch.exp(_kernel_val) for _kernel_val in kernel_vals]
        # return sum(kernel_val)
        return kernels
    
    def get_contrast_loss2(self,source,s_label):
        batch_size = source.shape[0]
        kernels = self.source_guassian_kernel2(source, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        if torch.sum(torch.isnan(sum(kernels))): return torch.tensor([0.]).cuda()
        s_label = s_label.contiguous().view(-1, 1)
        mask = torch.eq(s_label, s_label.T).float().cuda()
        t_loss=[]
        for kernel in kernels:
            # compute logits
            # anchor_dot_contrast = kernels
            anchor_dot_contrast = kernel
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).cuda(),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            non_zero_mask = mask.sum(1)!=0
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask[non_zero_mask] * log_prob[non_zero_mask]).sum(1) / mask[non_zero_mask].sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
            t_loss.append(loss)

        return sum(t_loss)/len(t_loss)

    def get_loss_cate(self, source, target, s_label, t_label):
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label.cpu().data.numpy()
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        indices = list(set(s_sca_label) & set(t_sca_label))
        for index in indices:
            si_batch_size = (s_sca_label==index).sum().item()
            ti_batch_size = (t_sca_label==index).sum().item()
            si = source[s_sca_label==index]
            ti = target[t_sca_label==index]
            kernels = self.guassian_kernel(si, ti, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            kernels /= (si_batch_size + ti_batch_size)**2
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            #2023-03-06把这个地方的数字修改了一下
            loss += torch.sum(TT) - torch.sum(ST) - torch.sum(TS)
            # loss += torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS)
            # loss += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))/(si_batch_size+ti_batch_size)
            # loss += (torch.sum(SS)/(si_batch_size**2) + torch.sum(TT)/(ti_batch_size**2) - torch.sum(ST)/(si_batch_size*ti_batch_size) - torch.sum(TS)/(si_batch_size*ti_batch_size))/(si_batch_size+ti_batch_size)

        loss/=len(indices)
        return loss

    def get_loss(self, source, target, s_label, t_label):
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label.cpu().data.numpy()
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        # indices = list(set(s_sca_label) & set(t_sca_label))
        indices = list(set(t_sca_label))
        for index in indices:
            si_batch_size = (s_sca_label==index).sum().item()
            ti_batch_size = (t_sca_label==index).sum().item()
            si = source[s_sca_label==index]
            ti = target[t_sca_label==index]
            kernels = self.guassian_kernel(si, ti,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            kernels /= (si_batch_size + ti_batch_size)**2
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            #2023-03-06把这个地方的数字修改了一下
            loss += torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS)
            # loss += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))/(si_batch_size+ti_batch_size)
            # loss += (torch.sum(SS)/(si_batch_size**2) + torch.sum(TT)/(ti_batch_size**2) - torch.sum(ST)/(si_batch_size*ti_batch_size) - torch.sum(TS)/(si_batch_size*ti_batch_size))/(si_batch_size+ti_batch_size)

        loss/=len(indices)
        return loss

    def cal_weight(self, s_label, t_label):
        source_batch_size = s_label.size()[0]
        target_batch_size = t_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = np.eye(self.class_num)[s_sca_label]
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, self.class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum


        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        # t_vec_label = np.eye(self.class_num)[t_sca_label]
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, self.class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        s_mask_arr = np.zeros((source_batch_size, self.class_num))
        s_mask_arr[:, index] = 1
        s_vec_label = s_vec_label * s_mask_arr

        t_mask_arr = np.zeros((target_batch_size, self.class_num))
        t_mask_arr[:, index] = 1
        t_vec_label = t_vec_label * t_mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)
        weight_ts = np.matmul(t_vec_label, s_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
            weight_ts = weight_ts / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
            weight_ts = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32'), weight_ts.astype("float32")

    def get_loss_w(self, source, target, s_label, t_label):
        source = source[s_label!=-1]
        s_label = s_label[s_label!=-1]
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = np.eye(self.class_num)[s_sca_label]
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, self.class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_vec_label = t_label.cpu().data.numpy()
        t_sca_prob = t_label.cpu().data.max(1)[0].numpy()
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()

        t_sum = np.sum(t_vec_label, axis=0).reshape(1, self.class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        indices = list(set(s_sca_label)&set(t_sca_label))

        for index in indices:
            target_mask = t_sca_label==index
            ti_batch_size = target_mask.sum().item()
            ti = target[target_mask]
            source_mask = s_sca_label==index
            si_batch_size = source_mask.sum().item()
            si = source[source_mask]
            weight_ss = np.matmul(s_vec_label[source_mask], s_vec_label[source_mask].T).astype('float32')
            weight_tt = np.matmul(t_vec_label[target_mask], t_vec_label[target_mask].T).astype('float32')
            weight_st = np.matmul(s_vec_label[source_mask], t_vec_label[target_mask].T).astype('float32')
            weight_ts = np.matmul(t_vec_label[target_mask], s_vec_label[source_mask].T).astype('float32')

            weight_ss = torch.from_numpy(weight_ss).cuda()
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()
            weight_ts = torch.from_numpy(weight_ts).cuda()

            kernels = self.guassian_kernel(si, ti, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            #(B1,B1),(B2,B2),(B1,B2),(B2,B1)
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            loss += torch.sum(weight_ss*SS) + torch.sum(weight_tt * TT) - torch.sum(weight_st * ST) - torch.sum(weight_ts * TS)
        loss/=len(indices)
        return loss

    def get_loss_lmmd(self, source, target, s_label, t_label):

        source = source[s_label!=-1]
        s_label = s_label[s_label!=-1]

        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st, weight_ts = self.cal_weight(s_label, t_label)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()
        weight_ts = torch.from_numpy(weight_ts).cuda()

        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        TS = kernels[batch_size:,:batch_size]

        loss += torch.sum(weight_ss * SS) + torch.sum(weight_tt * TT) - torch.sum(weight_st * ST) - torch.sum(weight_ts * TS)
        return loss

class dy_class_MMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(dy_class_MMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label,num_source_data):
        loss_source,loss_target = torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda()

        ss_sca_label = s_label[:num_source_data].cpu().data.numpy()
        source_s = source[:num_source_data]
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        indices = list(set(ss_sca_label) & set(t_sca_label))
        for index in indices:
            si_batch_size = (ss_sca_label==index).sum().item()
            ti_batch_size = (t_sca_label==index).sum().item()
            si = source_s[ss_sca_label==index]
            ti = target[t_sca_label==index]
            kernels = self.guassian_kernel(si, ti,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            loss_source += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))/(si_batch_size+ti_batch_size)
        loss_source /=len(indices)

        st_sca_label = s_label[num_source_data:].cpu().data.numpy()
        source_t = source[num_source_data:]
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        indices = list(set(st_sca_label) & set(t_sca_label))
        for index in indices:
            si_batch_size = (st_sca_label==index).sum().item()
            ti_batch_size = (t_sca_label==index).sum().item()
            si = source_t[st_sca_label==index]
            ti = target[t_sca_label==index]
            kernels = self.guassian_kernel(si, ti,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            loss_target += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))/(si_batch_size+ti_batch_size)

        loss_target/=len(indices)

        return loss_source,loss_target
    
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, max_epochs=30, lambda_u=75):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, max_epochs)

class WeightEMA(object):
    def __init__(self, model, ema_model, lr=1e-3,alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    features = input_list[0]
    source_batch_size = features[0].shape[0]
    target_batch_size = sum([f.shape[0] for f in features[1:]])
    feature = torch.concat(features,dim=0)
    softmax_output = torch.concat(input_list[1],dim=0).detach()
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    dc_target = torch.from_numpy(np.array([[1]] * source_batch_size + [[0]] * target_batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[source_batch_size:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:source_batch_size] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 



class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def cross_entropy_loss(preds, target, reduction):
    logp = F.log_softmax(preds, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

def onehot_encoding(labels, n_classes):
    return torch.zeros(labels.size(0), n_classes).to(labels.device).scatter_(
        dim=1, index=labels.view(-1, 1), value=1)
    
def label_smoothing(preds, targets,epsilon=0.1):
	#preds为网络最后一层输出的logits
	#targets为未one-hot的真实标签
    n_classes = preds.size(1)
    device = preds.device
    
    onehot = onehot_encoding(targets, n_classes).float().to(device)
    targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(
        device) * epsilon / n_classes
    loss = cross_entropy_loss(preds, targets, reduction="mean")
    return loss

def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
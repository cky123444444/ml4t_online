import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_log1p(x):
    x = x.to(torch.float32)
    x = torch.clamp(x, min=0.0)  # 等价于 tf.math.maximum
    return torch.log1p(x)


class Target(nn.Module):
    def __init__(self, name, output, label, mask):
        super().__init__()
        self.name = name
        self.output = output
        self.mask = mask
    
    def transform(self, label):
        for transform in self.transforms:
            if transform == None:
                continue
            elif transform == 'log1p':
                label = torch_log1p(label)
            else:
                print('error')
        return label
    
    def get_label(self, label):
        return label
    


class RegularizationTarget(nn.Module):
    def __init__(self, name, output, label, mask):
        super().__init__()
        self.name = name
        self.output = output
        self.mask = mask
    
    def transform(self, label):
        for transform in self.transforms:
            if transform == None:
                continue
            elif transform == 'log1p':
                label = torch_log1p(label)
            else:
                print('error')
        return label
    
    def get_label(self, label):
        return label
    
def get_pred_and_loss(logit, label, loss_weight, loss_type, mask=None, class_weight=None):
    logit = torch.clamp(logit, min=-50, max=50)
    label = label.float().unsqueeze(1)
    batch_size = logit.shape[0]
    valid_batch_size = mask.float().sum() if mask is not None else torch.tensor(batch_size, dtype=torch.float32)
    if loss_type == 'logloss':
        pred = F.sigmoid(logit)
        # print("pred: ", pred)
        # print("logtis: ", logit.shape, logit)
        # print("label:", label)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # 保留逐样本 loss，方便加 mask
        loss = loss_fn(logit, label)  # shape: (batch,)
        #print("pre loss: ", logit.shape, logit)
        
    elif loss_type == 'mse':
        pred = F.leaky_relu(logit)
        pred = logit
        loss = 0.5 * torch.square(label - logit)
        #print("pre loss: ", logit.shape, logit)

    elif loss_type == 'distill_softmax':
        #使用ds时，输出tower维度应该和bucket_num一致
        bucket_num = 50
        max_label = 1.0
        label_bounds = (torch.linspace(0, max_label, bucket_num)).unsqueeze(0)
        real_label = label.unsqueeze(1)
        temperature = torch.sqrt(real_label) + 1e-6
        real_label = real_label.repeat(1, bucket_num)
        temperature = temperature.repeat(1, bucket_num)
        probs = torch.exp(-torch.abs(real_label - label_bounds) / temperature)
        probs_norm = torch.sum(probs, dim=1, keepdim=True)
        label_distill = probs / probs_norm
        log_logit = F.log_softmax(logit, dim=1)
        loss = -torch.sum(label_distill * log_logit, dim=1)
        pred_probs = torch.softmax(logit, dim=1)
        pred = torch.sum(label_bounds * pred_probs, dim=-1)

    elif loss_type == 'cross_entropy':
        label = label.long().squeeze()
        if not isinstance(class_weight, torch.Tensor) or class_weight.shape[0] != logit.shape[1]:
            loss = F.cross_entropy(logit, label, reduction='none')
        else:
            loss = F.cross_entropy(logit, label, weight=class_weight, reduction='none')
        pred = F.softmax(logit, dim=1)

    elif loss_type == 'focal_loss':
        label = label.long().squeeze()
        gamma = 2.0   # Adjust as needed  
        ce_loss = F.cross_entropy(logit, label, reduction='none')
        pred = F.softmax(logit, dim=1)
        true_probs = pred.gather(1, label.unsqueeze(1)).squeeze()
        focal_weight = class_weight[label] * (1 - true_probs) ** gamma
        extra_weight = 1.05  # 对于 label=0 的额外权重
        focal_weight = torch.where(label == 0, focal_weight * extra_weight, focal_weight)
        loss = focal_weight * ce_loss
        
    else:
        raise ValueError('Unknown loss_type: {}'.format(loss_type))

    loss = loss * loss_weight
    #print("weighted loss: ", loss.shape, loss)
    # print('valid_batch_size.shape', valid_batch_size)
    # print('ones_like',torch.ones_like(valid_batch_size, dtype=torch.float32))
    # print('loss.shape', loss.shape)
    if mask is not None:
        loss = loss.squeeze() * mask
        #print('mask:', mask.shape, mask)
    #print("proceed loss: ", loss.shape, loss)
    loss = torch.sum(loss)
    loss = loss / torch.maximum(valid_batch_size, torch.ones_like(valid_batch_size, dtype=torch.float32))
    # if loss_type=='cross_entropy':
    #     print("batch size:", batch_size)
    #     print("valid batch size:", valid_batch_size.shape, valid_batch_size)
    #     print("loss:", loss.shape, loss)
    return pred, loss


def get_tarreg_pred_and_loss(name, logit, label, loss_weight, eps):
    valid_batch_size = torch.tensor(logit[0].shape[0], dtype=torch.float32)
    t_pred, y0_pred, y1_pred = logit[0], logit[1], logit[2]
    t_true = label[0].float().unsqueeze(1)
    y_true = label[1].float().unsqueeze(1)
    # mask = tf.logical_and(y_true > -10000, y_true < 1000)
    # valid_batch_size = tf.reduce_sum(tf.cast(mask, tf.float32))
    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred  # pred for 'real' treatment
    t_pred = (t_pred + 0.01) / 1.02
    h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
    y_pert = y_pred + eps(h)
    # if name == 'return_tar_loss':
    #     print("t_pred: ", t_pred.shape, t_pred)
    #     print("y0_perd: ", y0_pred.shape, y0_pred)
    #     print("y1_perd: ", y1_pred.shape, y1_pred)
    #     print("t_true: ", t_true.shape, t_true)
    #     print("y_true: ", y_true.shape, y_true)
    #     print("y_pred: ", y_pred.shape, y_pred)
    #     print("y_pert: ", y_pert.shape, y_pert)
        
    loss = torch.square(y_true - y_pert)
    # if name == 'return_tar_loss':
    #     print("loss:", loss.shape)
    loss = loss * loss_weight
    # if name == 'return_tar_loss':
    #     print("loss*loss_weight:", loss.shape)
    loss = torch.sum(loss)
    # if name == 'return_tar_loss':
    #     print("loss sum:", loss.shape, loss)
    loss = loss / torch.maximum(valid_batch_size, torch.ones_like(valid_batch_size, dtype=torch.float32))
    # if name == 'return_tar_loss':
    #     print("loss mean:", loss.shape, loss)
    #     print('valid_batch_size:', valid_batch_size)
    return y_pred, loss






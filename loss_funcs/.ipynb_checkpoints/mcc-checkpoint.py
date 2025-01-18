from loss_funcs.mmd import MMDLoss
from loss_funcs.adv import LambdaSheduler
import torch
import numpy as np
import torch.nn.functional as F


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class MYLoss(MMDLoss, LambdaSheduler):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                 gamma=1.0, max_iter=1000, **kwargs):
        '''
        Local MMD
        '''
        super(MYLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
        super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
        self.num_class = num_class
        self.temperature = 2.5

    def forward(self, source, target, source_label, target_logits,target_classification):
        logits = target_classification
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        
        
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")
        
        elif self.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
            weight_ss = torch.from_numpy(weight_ss).cuda() # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.guassian_kernel(source, target,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            loss = torch.Tensor([0]).cuda()
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]

            loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
            # Dynamic weighting
            lamb = self.lamb()
            self.step()
            loss = loss * lamb
            
        return [mcc_loss, loss]


    
    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label] # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum # label ratio

        # Pseudo label
        target_label = target_logits.cpu().data.max(1)[1].numpy()

        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits / target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)
        set_t = set(target_label)
        count = 0
        for i in range(self.num_class): # (B, C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1) # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1) # (B, 1)
                
                ss = np.dot(s_tvec, s_tvec.T) # (B, B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st     
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
        
        

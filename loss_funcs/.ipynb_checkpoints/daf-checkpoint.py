from loss_funcs.mmd import MMDLoss
from loss_funcs.adv import LambdaSheduler
import torch
import numpy as np
import torch.nn.functional as F


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


class DAFLoss(MMDLoss, LambdaSheduler):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                 gamma=1.0, max_iter=1000, **kwargs):
        super(DAFLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
        super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
        self.num_class = num_class
        self.temperature = 2.5
        self.g_l, self.p_l = 0, 0
        self.dynamic_factor = 0.5

    def forward(self, source, target, source_label, target_logits, target_classification):
        logits = target_classification
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0),
                                          predictions)  # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes

        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")

        elif self.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
            weight_ss = torch.from_numpy(weight_ss).cuda()  # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.guassian_kernel(source, target,
                                           kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            global_loss = torch.Tensor([0]).cuda()
            partial_loss = torch.Tensor([0]).cuda()

            if torch.sum(torch.isnan(sum(kernels))):
                return global_loss, partial_loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]
            TS = kernels[batch_size:, :batch_size]

            global_loss += torch.mean(SS + TT - ST - TS)
            partial_loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
            # Dynamic weighting
            lamb = self.lamb()
            self.step()
            global_loss = global_loss * lamb
            partial_loss = partial_loss * lamb
            self.g_l = self.g_l + 2 * (1 - 2 * global_loss.cpu().item())
            self.p_l = self.p_l + 2 * (1 - 2 * partial_loss.cpu().item())
            loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * partial_loss

        return (mcc_loss + loss) * 0.5

    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label]  # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum  # label ratio

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
        for i in range(self.num_class):  # (B, C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)  # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1)  # (B, 1)

                ss = np.dot(s_tvec, s_tvec.T)  # (B, B)
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

    def update_dynamic_factor(self, epoch_length):
        if self.g_l == 0 and self.p_l == 0:
            self.dynamic_factor = 0.5
        else:
            self.g_l = self.g_l / epoch_length
            self.p_l = self.p_l / epoch_length
            self.dynamic_factor = 1 - self.g_l / (self.g_l + self.p_l)
        self.g_l, self.p_l = 0, 0



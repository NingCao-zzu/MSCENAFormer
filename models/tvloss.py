import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        # 初始化为1，但会在训练过程中进行调整
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return self.weight * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

import torch.nn as nn
import numpy as np
import torch
from ..backbone.IFSxMLP1 import net1
from ..backbone.IFSxMLP2 import net2

class NetWrapper(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, l_f, e_l_f, g_f, e_g_f):
        representation1 = self.net1(l_f, g_f)
        representation2 = self.net2(e_l_f, e_g_f)

        return representation1 ,representation2

## CIELF model
class CIELF(nn.Module):
    def __init__(self, T=0.1):
        super(CIELF, self).__init__()

        self.net = NetWrapper(net1=net1(), net2=net2())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T))

    def forward(self, l_f, e_l_f, g_f, e_g_f):
        n = l_f.shape[0]

        image1_features, image2_features = self.net(l_f, e_l_f, g_f, e_g_f)

        # normalized features
        image1_features = image1_features / image1_features.norm(dim=1, keepdim=True)
        image2_features = image2_features / image2_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image1 = logit_scale * image1_features @ image2_features.t()
        logits_per_image2 = logits_per_image1.t()

        labels = torch.arange(n).to(logits_per_image1.device)

        loss1 = nn.CrossEntropyLoss()(logits_per_image1, labels)
        loss2 = nn.CrossEntropyLoss()(logits_per_image2, labels)

        loss = (loss1 + loss2) / 2

        return loss

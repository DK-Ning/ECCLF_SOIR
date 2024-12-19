import torch.nn as nn
import numpy as np
import torch
from ..backbone.TiT1 import TiT1
from ..backbone.TiT2 import TiT2

class NetWrapper(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, im_1, im_1_b, huffman, huffman_b):
        representation1 = self.net1(im_1, huffman)
        representation2 = self.net2(im_1_b, huffman_b)

        return representation1 ,representation2

## ECCLF model
class ECCLF(nn.Module):
    def __init__(self, T=0.1):
        super(CLIP, self).__init__()

        self.net = NetWrapper(net1=TiT1(), net2=TiT2())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T))

    def forward(self, im_1, im_1_b, huffman, huffman_b):
        n = im_1.shape[0]

        image_features, text_features = self.net(im_1, im_1_b, huffman, huffman_b)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(n).to(logits_per_image.device)

        loss1 = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss2 = nn.CrossEntropyLoss()(logits_per_text, labels)

        loss = (loss1 + loss2) / 2

        return loss

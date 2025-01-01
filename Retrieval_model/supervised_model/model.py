import torch.nn as nn

## supervised model
class supervised_net(nn.Module):
    def __init__(self, net, out_dim=100):
        super(supervised_net, self).__init__()
        self.net = net
        self.out_dim = out_dim
        self.bn = nn.BatchNorm1d(256)
        self.head = nn.Linear(256,out_dim)

    def forward(self, l_f, e_l_f, g_f, e_g_f, label=None):

        image_features1, image_features2 = self.net(l_f, e_l_f, g_f, e_g_f)
        image_features1 = self.bn(image_features1)
        image_features2 = self.bn(image_features2)

        if label is not None:
            out1 = self.head(image_features1)
            out2 = self.head(image_features2)
            return image_features1, out1, image_features2, out2
        else:
            return image_features1, image_features2



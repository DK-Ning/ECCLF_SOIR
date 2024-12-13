import torch.nn as nn

## supervised model
class supervised_net(nn.Module):
    def __init__(self, net, out_dim=100):
        super(supervised_net, self).__init__()
        self.net = net
        self.out_dim = out_dim
        self.bn = nn.BatchNorm1d(256)
        self.head = nn.Linear(256,out_dim)

    def forward(self, im_1, im_1_b, huffman, huffman_b, label=None):

        image_features, text_features = self.net(im_1, im_1_b, huffman, huffman_b)
        image_features = self.bn(image_features)
        text_features = self.bn(text_features)

        if label is not None:
            out1 = self.head(image_features)
            out2 = self.head(text_features)
            return image_features, out1, text_features, out2
        else:
            return image_features, text_features



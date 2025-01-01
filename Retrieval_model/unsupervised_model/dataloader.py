from torch.utils.data import Dataset
import torch

class DataPair(Dataset):

    def __init__(self, local_feature, edge_loacal_feature, global_feature, edge_global_feature, transform=None):
        self.local_feature = local_feature
        self.edge_loacal_feature = edge_loacal_feature
        self.global_feature = global_feature
        self.edge_global_feature = edge_global_feature
        self.transform = transform

    def __getitem__(self, index):
        local_f = self.local_feature[index]
        edge_loacal_f = self.edge_loacal_feature[index]
        global_f = torch.tensor(self.global_feature[index])
        edge_global_f = torch.tensor(self.edge_global_feature[index])

        if self.transform is not None:
            local_f_aug = self.transform(local_f)
            edge_loacal_f_aug = self.transform(edge_loacal_f)

        return local_f_aug, edge_loacal_f_aug, global_f, edge_global_f

    def __len__(self):
        return len(self.local_feature)

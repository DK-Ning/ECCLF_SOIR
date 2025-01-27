import argparse
import torch
import datetime
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from ..dataAug import Exchange_Block, Concat_Prior_to_Last
from dataloader import DataPair
from ..data_utils import split_data
import torch.nn.functional as F
from ..schedule import get_cosine_schedule_with_warmup
from model import CIELF

parser = argparse.ArgumentParser(description='Train unsupervised on CIELF')
args = parser.parse_args('')

## set training parameter
args.lr = 1e-3
args.weight_decay = 5e-5
args.type = 'Corel10K'
args.epochs = 300

# load trian data
train_local_feature, train_edge_local_feature, test_local_feature, test_edge_local_feature, train_global_feature, train_edge_global_feature, \
        test_global_feature, test_edge_global_feature, train_label, test_label = split_data()

train_transform = transforms.Compose([
    Exchange_Block(0.3),
    Concat_Prior_to_Last(0.3),
    transforms.ToTensor()])

test_transform = transforms.Compose([transforms.ToTensor()])

train_data = DataPair(local_feature=train_local_feature, edge_loacal_feature=train_edge_local_feature, global_feature=train_global_feature,edge_global_feature=train_edge_global_feature, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)

test_data = DataPair(local_feature=test_local_feature,edge_loacal_feature=test_edge_local_feature,global_feature=test_global_feature,edge_global_feature=test_edge_global_feature, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=20, shuffle=False, num_workers=20, pin_memory=True)


# train for one epoch
def train(net, data_loader, train_optimizer, epoch, scheduler, args):
    net.train()
    scheduler.step()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for l_f1, l_f2, g_f1, g_f2 in train_bar:
        l_f1, l_f2, g_f1, g_f2 = l_f1.cuda(non_blocking=True), l_f2.cuda(non_blocking=True), g_f1.cuda(non_blocking=True), g_f2.cuda(non_blocking=True)

        loss = net(l_f1, l_f2, g_f1, g_f2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch+1, args.epochs,
                                                                                          train_optimizer.param_groups[
                                                                                              0]['lr'],
                                                                                          total_loss / total_num))
    return total_loss / total_num

## test stage
def test(net, test_loader, test_label):
    net.eval()
    feature_bank = []
    train_bar = tqdm(test_loader)
    with torch.no_grad():
        for l_f1, l_f2, g_f1, g_f2 in train_bar:
            l_f1, l_f2, g_f1, g_f2 = l_f1.cuda(non_blocking=True), l_f2.cuda(non_blocking=True), g_f1.cuda(non_blocking=True), g_f2.cuda(non_blocking=True)

            feature1, feature2 = net(l_f1, l_f2, g_f1, g_f2)
            feature1 = F.normalize(feature1, dim=1)
            feature2 = F.normalize(feature2, dim=1)
            feature = 0.6*feature1 + 0.4*feature2
            feature_bank.append(feature)

        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        feature_labels = torch.tensor(test_label, device=feature_bank.device)
        average_precision_li = []
        for idx in range(feature_bank.size(0)):
            query = feature_bank[idx].expand(feature_bank.shape)

            label = feature_labels[idx]
            sim = F.cosine_similarity(feature_bank, query)
            _, indices = torch.topk(sim, 100)
            match_list = feature_labels[indices] == label
            pos_num = 0
            total_num = 0
            precision_li = []

            for item in match_list[1:]:
                if item == 1:
                    pos_num += 1
                    total_num += 1
                    precision_li.append(pos_num / float(total_num))
                else:
                    total_num += 1
            if precision_li == []:
                average_precision_li.append(0)
            else:
                average_precision = np.mean(precision_li)
                average_precision_li.append(average_precision)
        mAP = np.mean(average_precision_li)
        print('image test mAP:',mAP)


if __name__ == '__main__':

    model = CIELF().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=20,
                                                 num_training_steps=args.epochs)

    epoch_start = 0
    # training loop
    now_time = datetime.datetime.now()
    print(now_time)

    loss_values = []
    for epoch in range(epoch_start, args.epochs):
       train_loss = train(model, train_loader, optimizer, epoch, scheduler, args)
       loss_values.append(train_loss)

    with open('unsupervised_loss_values.pkl', 'wb') as file:
       pickle.dump(loss_values, file)

    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'unsupervised_model_last.pth')


    now_time = datetime.datetime.now()
    print(now_time)
    ## test
    test(model.net, test_loader, test_label)

    now_time = datetime.datetime.now()
    print(now_time)



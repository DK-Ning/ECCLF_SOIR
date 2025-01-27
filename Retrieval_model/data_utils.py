import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_data():
    ## load plain-cipherimage local features
    folder_path = '../data/features/local_feature'
    file_list = os.listdir(folder_path)
    file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
    all_arrays = []
    for file_name in file_list:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            array = np.load(file_path, allow_pickle=True).astype(np.float32)
            all_arrays.append(array)
    local_features = np.concatenate(all_arrays, axis=0)

    ## load edge-cipherimage local features
    folder_path2 = '../data/edge_features/local_feature'
    file_list2 = os.listdir(folder_path2)
    file_list2 = sorted(file_list2, key=lambda x: int(x.split('.')[0]))
    all_arrays2 = []
    for file_name in file_list2:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path2, file_name)
            array2 = np.load(file_path, allow_pickle=True).astype(np.float32)
            all_arrays2.append(array2)
    edge_local_features = np.concatenate(all_arrays2, axis=0)

    ## load plain-cipherimage global features
    folder_path3 = '../data/features/global_feature'
    file_list3 = os.listdir(folder_path3)
    file_list3 = sorted(file_list3, key=lambda x: int(x.split('.')[0]))
    all_arrays3 = []
    for file_name in file_list3:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path3, file_name)
            array3 = np.load(file_path, allow_pickle=True).astype(np.float32)
            all_arrays3.append(array3)
    global_features = np.stack(all_arrays3, axis=0)

    ## load edge-cipherimage global features
    folder_path4 = '../data/edge_features/global_feature'
    file_list4 = os.listdir(folder_path4)
    file_list4 = sorted(file_list4, key=lambda x: int(x.split('.')[0]))
    all_arrays4 = []
    for file_name in file_list4:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path4, file_name)
            array4 = np.load(file_path, allow_pickle=True).astype(np.float32)
            all_arrays4.append(array4)
    edge_global_features = np.stack(all_arrays4, axis=0)

    label = []
    for i in range(100):
        for j in range(100):
            label.append(i)

    train_local_feature, test_local_feature, train_label, test_label = train_test_split(local_features, label, test_size=0.3, stratify=label, random_state=20240)
    train_edge_local_feature, test_edge_local_feature, train_label, test_label = train_test_split(edge_local_features, label, test_size=0.3, stratify=label, random_state=20240)

    train_global_feature, test_global_feature, train_label, test_label = train_test_split(global_features, label, test_size=0.3, stratify=label, random_state=20240)
    train_edge_global_feature, test_edge_global_feature, train_label, test_label = train_test_split(edge_global_features, label, test_size=0.3, stratify=label, random_state=20240)

    return train_local_feature, train_edge_local_feature, test_local_feature, test_edge_local_feature, train_global_feature, train_edge_global_feature, \
        test_global_feature, test_edge_global_feature, train_label, test_label



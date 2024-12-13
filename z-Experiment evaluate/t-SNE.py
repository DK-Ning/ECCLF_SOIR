import os
import numpy as np
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

## draw t-SNE

random.seed(6)
np.random.seed(6)

save_folder = 'data/image_features'
num_classes = 100  # total category
num_selected_classes = 10  # Number of categories selected for total category
num_samples_per_class = 30  # Number of samples selected for each category
feature_dim = 100  # Dimensions of feature

# random select category
selected_classes = random.sample(range(num_classes), num_selected_classes)

selected_features = []
selected_labels = []

for class_idx in selected_classes:
    class_features = []
    class_labels = []
    for file_name in os.listdir(save_folder):
        if file_name.startswith(f"{class_idx}_"):
            file_path = os.path.join(save_folder, file_name)
            feature = np.load(file_path)
            class_features.append(feature)
            class_labels.append(class_idx)
    # random select samples for each category
    selected_indices = random.sample(range(len(class_features)), num_samples_per_class)
    selected_features.extend([class_features[idx] for idx in selected_indices])
    selected_labels.extend([class_labels[idx] for idx in selected_indices])

selected_features = np.array(selected_features)
selected_labels = np.array(selected_labels)


# Use PCA to reduce dimension
pca = PCA(n_components=50)
data_pca = pca.fit_transform(selected_features)

# Use t-SNE to further reduce dimension
tsne = TSNE(n_components=2)
embeddings = tsne.fit_transform(data_pca)

#  visualization
plt.figure(figsize=(6, 6))
for class_idx in range(num_selected_classes):
    idx = np.where(selected_labels == selected_classes[class_idx])[0]
    plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f"Class {class_idx}", alpha=1, s=5)
plt.legend(title='Classes', loc='upper right')
plt.axis('off')
plt.tight_layout()
plt.show()
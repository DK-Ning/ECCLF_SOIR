import os
import cv2
import numpy as np

def get_size(img):
    row, col, _ = img.shape
    img = img.astype(np.float64)
    target_row, target_col = (192, 128) if row >= col else (128, 192)

    pad_row = target_row - row if target_row > row else 0
    pad_col = target_col - col if target_col > col else 0

    img = np.pad(img, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')

    img2 = cv2.resize(img, (target_col, target_row))

    return img2
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        # Therefore PSNR have no importance.
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_average_psnr(folder1, folder2):
    psnr_values = []
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    for file1, file2 in zip(files1, files2):
        path1 = os.path.join(folder1, file1)
        path2 = os.path.join(folder2, file2)

        img1 = cv2.imread(path1)
        img1 = get_size(img1)
        img2 = cv2.imread(path2)
        img2 = get_size(img2)

        if img1 is None or img2 is None:
            print(f"Error reading {file1} or {file2}")
            continue

        psnr = calculate_psnr(img1, img2)
        psnr_values.append(psnr)

    average_psnr = np.mean(psnr_values)
    return average_psnr


folder1 = 'data/plainimages'
folder2 = 'data/cipheriamges'

## calculate average PSNR
average_psnr = calculate_average_psnr(folder1, folder2)
print(f"Average PSNR: {average_psnr}")

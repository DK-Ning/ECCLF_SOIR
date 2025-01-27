import datetime
import cv2
import multiprocessing as mul
from Encryption_algorithm.encryption_utils import loadImageFiles

## generate Edge-images with Sobel algorithm
def imggen(imageFile):
    img = cv2.imread(imageFile)
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)

    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    result = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imwrite(bit_path + imageFile.split("/")[-1], result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print(imageFile + ' ' + 'process success!')

imageFiles = loadImageFiles('../data/plainimages/*.jpg')
bit_path = '../data/edge_imgs/'

if __name__ == '__main__':

    now_time = datetime.datetime.now()
    print(now_time)

    pool = mul.Pool(10)
    rel = pool.map(imggen, imageFiles)

    now_time = datetime.datetime.now()
    print(now_time)
    print('finish')
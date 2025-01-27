import datetime
import random
import os
import copy
import cv2
import hashlib
from Encryption_algorithm.encryption_utils import yates_shuffle
from Encryption_algorithm.JPEG.jdcencColor import jdcencColor
from Encryption_algorithm.JPEG.zigzag import zigzag
from Encryption_algorithm.JPEG.jacencColor import jacencColor
from Encryption_algorithm.JPEG.Quantization import *
from Crypto.Cipher import AES
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor
from Encryption_algorithm.JPEG.invzigzag import invzigzag
from Encryption_algorithm.JPEG.rgbandycbcr import ycbcr2rgb, rgb2ycbcr
from Encryption_algorithm.JPEG.DCT import idctJPEG
from Encryption_algorithm.JPEG.Quantization import iQuantization


def encryption_each_component(image_component, keys, type, row, col, N, QF):
    allblock8 = np.zeros([8, 8, int(row * col / (8 * 8))])
    allblock8_number = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            t = image_component[m:m + N, n:n + N] - 128
            allblock8[:, :, allblock8_number] = t
            allblock8_number = allblock8_number + 1

    dc_values = []
    for i in range(0, allblock8_number):
        t = copy.copy(allblock8[:, :, i])
        t = cv2.dct(t)  # DCT
        temp = Quantization(t, type=type)  # Quantization
        dc_values.append(temp[0, 0])
    data = [i for i in range(0, len(dc_values))]
    dc_shuffle = yates_shuffle(data, keys)
    # DC系数置乱
    dc_values_permuted = copy.copy(dc_values)
    for i in range(0, len(dc_shuffle)):
        dc_values_permuted[i] = dc_values[dc_shuffle[i] - 1]
    dc_values = copy.copy(dc_values_permuted)
    del dc_values_permuted
    # Huffman coding
    dc = 0
    dccof = []
    accof = []
    for i in range(0, allblock8_number):
        t = copy.copy(allblock8[:, :, i])
        t = cv2.dct(t)  # DCT
        temp = Quantization(t, type=type)  # Quanlity
        permuted_dc = dc_values[i]

        if i == 0:
            dc = permuted_dc
            key_numbers, dc_component = jdcencColor(dc, type, keys)
            dccof = np.append(dccof, dc_component)
            keys = keys[key_numbers:]
        else:
            dc_diff = permuted_dc - dc
            key_numbers, dc_component = jdcencColor(dc_diff, type, keys)
            dccof = np.append(dccof, dc_component)
            dc = permuted_dc
            keys = keys[key_numbers:]

       # AC系数哈夫曼编码
        acseq = []
        aczigzag = zigzag(temp)
        eobi = 0
        for j in range(63, -1, -1):
            if aczigzag[j] != 0:
                eobi = j
                break
        if eobi == 0:
            acseq = np.append(acseq, [999])
        else:
            acseq = np.append(acseq, aczigzag[1: eobi + 1])
            acseq = np.append(acseq, [999])
        key_numbers, ac_component = jacencColor(acseq, type, keys)
        keys = keys[key_numbers:]
        accof = np.append(accof, ac_component)

    return dccof, accof

def encryption(Y, Cb, Cr, keyY, keyCb, keyCr, QF, N, row, col):
    # N: block size
    # QF: quality factor
    row = int(row)
    col = int(col)
    # Y component
    dccofY, accofY = encryption_each_component(Y, keyY, type='Y', row=row, col=col, N=N, QF=QF)
    ## Cb and Cr component
    dccofCb, accofCb = encryption_each_component(Cb, keyCb, type='Cb', row=int(row), col=int(col),
                                                 N=N, QF=QF)
    dccofCr, accofCr = encryption_each_component(Cr, keyCr, type='Cr', row=int(row), col=int(col),
                                                 N=N, QF=QF)
    accofY = accofY.astype(np.int8)
    dccofY = dccofY.astype(np.int8)
    accofCb = accofCb.astype(np.int8)
    dccofCb = dccofCb.astype(np.int8)
    accofCr = accofCr.astype(np.int8)
    dccofCr = dccofCr.astype(np.int8)
    return accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr

## 13-2加密
def get_size(img):
    row, col, _ = img.shape

    img = rgb2ycbcr(img)
    img = img.astype(np.float16)
    target_row, target_col = (192, 128) if row >= col else (128, 192)

    pad_row = target_row - row if target_row > row else 0
    pad_col = target_col - col if target_col > col else 0

    img = np.pad(img, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')

    img2 = cv2.resize(img, (target_col, target_row))
    row2, col2, _ = img2.shape

    Y = img2[:, :, 0]
    Cb = img2[:, :, 1]
    Cr = img2[:, :, 2]

    return img2, row2, col2, Y, Cb, Cr
def extract_histogram(Y, Cb, Cr):

    hist_y, _ = np.histogram(Y, bins=256, range=(0, 256))
    hist_cb, _ = np.histogram(Cb, bins=256, range=(0, 256))
    hist_cr, _ = np.histogram(Cr, bins=256, range=(0, 256))

    return hist_y, hist_cb, hist_cr

hash = hashlib.blake2b(digest_size=32)
def generate_hash(inp):
    hash.update(bytes(str(inp), encoding='utf-8'))
    res = hash.digest()
    return res

def generate_key_from_histogram(hist):

    hist_data = hist.astype(np.uint8).tobytes()
    key = generate_hash(hist_data)
    return key

def generate_key_stream_using_ctr_mode(key, length):
    nonce = b'\x00' * 8  # initialize vector
    counter = 0
    key_stream = b''

    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)

    while len(key_stream) < length:
        counter_bytes = counter.to_bytes(8, byteorder='little')
        encrypted_block = cipher.encrypt(counter_bytes)
        key_stream += encrypted_block
        counter += 1

    return key_stream[:length]

def deEntropy(acall, dcall, row, col, type, N=8, QF = 100):
    _, acarr = jacdecColor(acall, type)
    _, dcarr = jdcdecColor(dcall, type)
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)
    row = int(row)
    col = int(col)
    Eob = np.where(acarr == 999)
    Eob = Eob[0]
    count = 0
    kk = 0
    ind1 = 0
    xq = np.zeros([row, col])
    for m in range(0, row, N):
        for n in range(0, col, N):
            ac = acarr[ind1: Eob[count]]
            ind1 = Eob[count] + 1
            count = count + 1
            acc = np.append(dcarr[kk], ac)
            az = np.zeros(64 - acc.shape[0])
            acc = np.append(acc, az)
            temp = invzigzag(acc, 8, 8)
            temp = iQuantization(temp, QF, type)
            temp = idctJPEG(temp)
            xq[m:m + N, n:n + N] = temp + 128
            kk = kk + 1
    return xq

def Gen_cipher_images(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size):
    cipher_Y = deEntropy(acallY, dcallY, img_size[0], img_size[1], 'Y')
    cipher_cb = deEntropy(acallCb, dcallCb, img_size[0], img_size[1], 'U')
    cipher_cr = deEntropy(acallCr, dcallCr, img_size[0], img_size[1], 'V')
    cipherimage = np.dstack([cipher_Y, cipher_cb, cipher_cr])
    cipherimage = np.round(cipherimage)
    cipherimage = cipherimage.astype(np.uint8)
    cipherimage = ycbcr2rgb(cipherimage)
    merged = cv2.merge([cipherimage[:, :, 2], cipherimage[:, :, 1], cipherimage[:, :, 0]])

    return merged

def encrypt_image(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, row, col, Y, Cb, Cr = get_size(img)

    hist_y, hist_cb, hist_cr = extract_histogram(Y, Cb, Cr)

    key_y = generate_key_from_histogram(hist_y)
    key_cb = generate_key_from_histogram(hist_cb)
    key_cr = generate_key_from_histogram(hist_cr)

    key_len = 64*384

    keystream_y = generate_key_stream_using_ctr_mode(key_y, key_len)
    keystream_cb = generate_key_stream_using_ctr_mode(key_cb, key_len)
    keystream_cr = generate_key_stream_using_ctr_mode(key_cr, key_len)

    encryption_keyY = ''.join(format(byte, '08b') for byte in keystream_y)
    encryption_keyCb = ''.join(format(byte, '08b') for byte in keystream_cb)
    encryption_keyCr = ''.join(format(byte, '08b') for byte in keystream_cr)

    accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr = encryption(Y, Cb, Cr, encryption_keyY,
                                                                    encryption_keyCb,
                                                                    encryption_keyCr,
                                                                    QF,
                                                                    N=8, row=row, col=col)

    img_size = (row, col)
    a = Gen_cipher_images(dccofY, accofY, dccofCb, accofCb, dccofCr, accofCr, img_size)

    return a

def change_one_pixel(image):
    # random select a pixel to change
    H, W, C = image.shape
    x = random.randint(0, H-1)
    y = random.randint(0, W-1)
    c = random.randint(0, C-1)
    image[x, y, c] = random.randint(0, 255)
    return image

def calculate_npcr(image1, image2):
    H, W = image1.shape
    D = np.sum(image1 != image2)
    NPCR = (D / (H * W)) * 100
    return NPCR

def differential_attack_test(image_path):
    # read plain-image
    image = cv2.imread(image_path)

    # encrypt plain-image
    encrypted_image1 = encrypt_image(image)
    encrypted_image1 = cv2.cvtColor(encrypted_image1, cv2.COLOR_BGR2GRAY)
    # change a pixel
    modified_image = change_one_pixel(image.copy())

    # encrypt changed-image
    encrypted_image2 = encrypt_image(modified_image)
    encrypted_image2 = cv2.cvtColor(encrypted_image2, cv2.COLOR_BGR2GRAY)
    # calculate NPCR
    npcr_value = calculate_npcr(encrypted_image1, encrypted_image2)

    return npcr_value

now_time = datetime.datetime.now()
print(now_time)

QF = 100
image_directory = 'data/plainimages'
image_paths = [os.path.join(image_directory, f'34_{i}.jpg') for i in range(3401, 3501)]

npcr_values = []

for image_path in image_paths:
    if os.path.exists(image_path):
        npcr, uaci = differential_attack_test(image_path)
        npcr_values.append(npcr)

## calculate mean NPCR
average_npcr = np.mean(npcr_values)
print(f"Average NPCR: {average_npcr}%")

now_time = datetime.datetime.now()
print(now_time)

import json
import cv2
import copy
import datetime
import hashlib
from encryption_utils import yates_shuffle
from encryption_utils import loadImageFiles
from JPEG.rgbandycbcr import rgb2ycbcr
from JPEG.jdcencColor import jdcencColor
from JPEG.zigzag import zigzag
from JPEG.jacencColor import jacencColor
from JPEG.Quantization import *
from bianyuan_cipherimageRgbGenerate import Gen_cipher_images
import multiprocessing as mul
from Crypto.Cipher import AES

## encrypt each color component
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

## encrypt process
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

## generate inital key
def generate_key_from_histogram(hist):

    hist_data = hist.astype(np.uint8).tobytes()
    key = generate_hash(hist_data)

    return key

## generate encryption key
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


def main(imageFile):
    # read plain-image
    img = cv2.imread(imageFile)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, row, col, Y, Cb, Cr = get_size(img)

    hist_y, hist_cb, hist_cr = extract_histogram(Y, Cb, Cr)

    key_y = generate_key_from_histogram(hist_y)
    key_cb = generate_key_from_histogram(hist_cb)
    key_cr = generate_key_from_histogram(hist_cr)

    keystream_y = generate_key_stream_using_ctr_mode(key_y, key_len)
    keystream_cb = generate_key_stream_using_ctr_mode(key_cb, key_len)
    keystream_cr = generate_key_stream_using_ctr_mode(key_cr, key_len)

    encryption_keyY = ''.join(format(byte, '08b') for byte in keystream_y)
    encryption_keyCb = ''.join(format(byte, '08b') for byte in keystream_cb)
    encryption_keyCr = ''.join(format(byte, '08b') for byte in keystream_cr)

    # save keys
    key = {'inital_keyY': key_y, 'inital_keyCb':key_cb, 'inital_keyCr':key_cr}

    with open(keys_path + imageFile.split("/")[-1].split('.')[0] + '.txt', 'w') as file:
        json.dump(key, file)

    accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr = encryption(Y, Cb, Cr, encryption_keyY,
                                                                    encryption_keyCb,
                                                                    encryption_keyCr,
                                                                    QF,
                                                                    N=8, row=row, col=col)

    img_size = (row, col)
    # save code coefficients bitstream
    np.save(bit_path + imageFile.split("/")[-1].split('.')[0] + '.npy',
            {'accofY': accofY, 'dccofY': dccofY, 'accofCb': accofCb, 'dccofCb': dccofCb, 'accofCr': accofCr,
             'dccofCr': dccofCr, 'size': img_size})

    print(imageFile + ' ' + 'process success!')

    Gen_cipher_images(dccofY, accofY, dccofCb, accofCb, dccofCr, accofCr, img_size, imageFile)


QF = 100
key_len = 64 * 384

imageFiles = loadImageFiles('../data/bianyuan_img/*.jpg')
bit_path = '../data/edge_JPEGBitStream/'
keys_path = '../data/edge_keys/'

## image encryption
if __name__ == '__main__':

    now_time = datetime.datetime.now()
    print(now_time)

    pool = mul.Pool(10)
    rel = pool.map(main, imageFiles)

    now_time = datetime.datetime.now()
    print(now_time)
    print('finish')


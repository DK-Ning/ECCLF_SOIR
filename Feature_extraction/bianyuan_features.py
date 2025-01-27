import glob
import datetime
import numpy as np
from local_feature import extract_local_feature
from Encryption_algorithm.encryption_utils import loadEncBit
from global_feature import extract_global_feature
import multiprocessing as mul

def main(path):

    bitstream = loadEncBit(path).item()  # load encrypted bitstream

    # extract local features
    local_feature = extract_local_feature(bitstream['dccofY'], bitstream['accofY'], bitstream['dccofCb'],
                                                  bitstream['accofCb'], bitstream['dccofCr'], bitstream['accofCr'], bitstream['size'])
    np.save("../data/edge_features/local_feature/" + path.split('/')[-1].split('.')[0] + ".npy", local_feature)

    # extract global features
    global_feature = extract_global_feature(bitstream['dccofY'], bitstream['accofY'], bitstream['dccofCb'],
                                            bitstream['accofCb'], bitstream['dccofCr'], bitstream['accofCr'])
    np.save("../data/edge_features/global_feature/" + path.split('/')[-1].split('.')[
        0] + ".npy", global_feature)

    print(path + ' ' + 'process success!')

if __name__ == '__main__':
    bit_path = '../data/edge_JPEGBitStream/*.npy'
    bitFiles = glob.glob(bit_path)
    now_time = datetime.datetime.now()
    print(now_time)
    pool = mul.Pool(10)
    rel = pool.map(main, bitFiles)
    now_time = datetime.datetime.now()
    print(now_time)
    print('finish')

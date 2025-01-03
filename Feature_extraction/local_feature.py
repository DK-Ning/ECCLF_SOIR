import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor

def local_feature(accof, dccof, row, col, type, N=8):
    _, acarr = jacdecColor(accof, type)
    _, dcarr = jdcdecColor(dccof, type, 'E')
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)
    retFeature = np.zeros((int(row*col/64), 64))
    blockIndex = 0
    Eob = np.where(acarr == 999)[0]
    count = 0  # Eob position
    dc_idx = 0
    ac_idx = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            ac = acarr[ac_idx: Eob[count]]
            ac_idx = Eob[count] + 1
            count = count + 1
            acc = np.append(dcarr[dc_idx], ac)
            dc_idx = dc_idx + 1
            az = np.zeros(64 - acc.shape[0])
            acc = np.append(acc, az)

            # acc length
            coePosition = 0
            # non-zero
            for coefficient in acc:
                if coefficient == 0:
                    retFeature[blockIndex, coePosition] = 0
                else:
                    tmp = bin(int(coefficient))
                    coeLen = len(tmp)
                    if tmp[0] == '-':
                        coeLen -= 3
                    else:
                        coeLen -= 2
                    retFeature[blockIndex, coePosition] = coeLen
                coePosition += 1

            blockIndex += 1
    if type == 'Y':
        return retFeature
    else:
        return retFeature[:, :32]

def extract_local_feature(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size):
    featureAll = []

    featureY = local_feature(acallY, dcallY, int(img_size[0]), int(img_size[1]), 'Y')
    featureCb = local_feature(acallCb, dcallCb, int(img_size[0]), int(img_size[1]), 'Cb')
    featureCr = local_feature(acallCr, dcallCr, int(img_size[0]), int(img_size[1]), 'Cr')
    featureAll.append(np.concatenate([featureY, featureCb, featureCr], axis=1).astype(np.int8))
    return featureAll


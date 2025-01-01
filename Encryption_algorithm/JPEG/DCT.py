import cv2

def dctJPEG(block):
    return cv2.dct(block)

def idctJPEG(block):
    return cv2.idct(block)


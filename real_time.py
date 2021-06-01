import cv2
import pickle
import numpy as np
import torch
import torch.nn
import torch.optim
from model import CNN
from collections import Counter
import time

PATH = 'model.pth'
IMG_SIZE = 28
labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

predictions = []

model = torch.load(PATH)
model.eval()


def make_predictions(frame):
    pred = model(frame)
    return pred


def time_series(last_preds):
    dict_freq = Counter(last_preds)
    max_key = max(dict_freq, key=lambda k: dict_freq[k])
    return max_key


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype("float32")

    #frame -= mean
    #img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    np_arr = np.array(frame)
    #print(np_arr)
    resize_t = torch.Tensor(np_arr).view(-1, 1, 28, 28)
    pred = make_predictions(resize_t)
    pred = pred.tolist()
    #print(labels_map[pred.index(max(pred))])
    predictions.append(labels_map[pred.index(max(pred))])
    if len(predictions) > 201:
        print(time_series(predictions[-200:]))

    #print(labels_map[pred[pred.argmax()]])


    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break


cv2.destroyWindow("preview")

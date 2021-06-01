import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import csv
import torch
from model import CNN


REBUILD_DATA = False


class Fashion:
    IMG_SIZE = 28
    BAG = "Zalando_images/bag"
    BOOT = "Zalando_images/boot"
    DRESS = "Zalando_images/dress"
    JACKET = "Zalando_images/jackets"
    PULLI = "Zalando_images/pulli"
    SANDAL = "Zalando_images/sandel"
    SHIRT = "Zalando_images/shirt"
    SNEAKER = "Zalando_images/sneaker"
    TOP = "Zalando_images/t-shirt_top"
    TROUSER = "Zalando_images/trousers"
    TESTING = "PetImages/Testing"
    LABELS = {BAG: 0, BOOT: 1, DRESS: 2, JACKET: 3, PULLI: 4, SANDAL: 5, SHIRT: 6, SNEAKER: 7, TOP: 8, TROUSER: 9}
    training_data = []

    # cat_count = 0
    # dog_count = 0

    def make_training_data(self):

        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(10)[self.LABELS[label]]])

                        print(type(self.training_data[0]))

                        # if label == self.CATS:
                        #     self.cat_count += 1
                        # elif label == self.DOGS:
                        #     self.dog_count += 1

                    except Exception as e:
                        #print(label, f, str(e))
                        pass
        with open('parrot.pkl', 'wb') as f:
            pickle.dump(self.training_data, f)
        #np.random.shuffle(self.training_data)


        #pickle.dumps(self.training_data, ("training_data.npy",))
        # print(f"Cats: {self.cat_count}")
        # print(f"Dogs: {self.dog_count}")


if REBUILD_DATA:
    test = Fashion()
    test.make_training_data()

with open('parrot.pkl', 'rb') as f:
    my_new_list = pickle.load(f)

model = torch.load('model.pth')


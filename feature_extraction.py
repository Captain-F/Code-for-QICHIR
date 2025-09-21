from tensorflow.keras.applications import VGG19, ResNet50, Xception
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Reshape
import numpy as np
from utils import read_pickle, save_pickle
from sklearn.model_selection import train_test_split


def feature_extract(name):
    if name == 'ns':
        img_ns_data = read_pickle("path/to/ns.pickle")
        ns_imgs = np.concatenate((img_ns_data["bj"]["img"], img_ns_data["dwns"]["img"], \
                                   img_ns_data["hsnr"]["img"], img_ns_data["nrz"]["img"]), axis=0)
        labs = img_ns_data["bj"]["labs"] * [0] + img_ns_data["dwns"]["labs"] * [1]\
               + img_ns_data["hsnr"]["labs"] * [2] + img_ns_data["nrz"]["labs"] * [3]

        base_model = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        head_model = base_model.output
        pre_model = Model(inputs=base_model.input, outputs=head_model)

        imgs = pre_model.predict(np.array(ns_imgs), batch_size=8)

    elif name == 'nh':
        img_nh_data = read_pickle("path/to/nh.pickle")
        nh_imgs = np.concatenate((img_nh_data["thw"]["img"], \
                                  img_nh_data["ylq"]["img"], img_nh_data["zxz"]["img"]), axis=0)
        labs = len(img_nh_data["thw"]["labs"]) * [0] \
               + len(img_nh_data["ylq"]["labs"]) * [1] + len(img_nh_data["zxz"][:210]) * [2]

        base_model = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        head_model = base_model.output
        pre_model = Model(inputs=base_model.input, outputs=head_model)

        nh_imgs_ = pre_model.predict(np.array(nh_imgs), batch_size=8)

    elif name == 'xiu':
        img_xiu_data = read_pickle("path/to/xiu.pickle")
        xiu_imgs = np.concatenate((img_xiu_data["sux"]["img"], img_xiu_data["xx"]["img"], \
                                   img_xiu_data["shux"]["img"], img_xiu_data["yx"]["img"]), axis=0)
        labs = img_xiu_data["sux"]["labs"] * [0] + img_xiu_data["xx"]["labs"] * [1]\
               + img_xiu_data["shux"]["labs"] * [2] + img_xiu_data["yx"]["labs"] * [3]

        base_model = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        head_model = base_model.output
        pre_model = Model(inputs=base_model.input, outputs=head_model)

        imgs = pre_model.predict(np.array(xiu_imgs), batch_size=8)

    trainX, test_valid_X, trainY, test_valid_Y = \
        train_test_split(imgs, labs, test_size=0.3, random_state=10)

    testX, validX, testY, validY = \
        train_test_split(imgs, imgs, test_size=0.5, random_state=10)

    return  trainX, trainY, testX, testY, validX, validY
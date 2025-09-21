from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import re
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from utils import read_pickle, save_pickle


def sort_p(paths):
    paths_ = sorted(paths, key=lambda x: int(re.split(r'[\\.]', str(x))[-2]))
    return paths_


def get_nh_data():
    thw_img_p = sort_p(list(Path(r'path/to/nh/thw').glob('*/*.jpg')))
    ylq_img_p = sort_p(list(Path(r'path/to/nh/ylq').glob('*/*.jpg')))
    zxz_img_p = sort_p(list(Path(r'path/to/nh/zxz').glob('*/*.jpg')))

    thw_img_arr = [cv2.imread(str(p)) for p in thw_img_p]
    thw_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in thw_img_arr]
    thw_img = np.array(thw_img_res).astype(np.float32) / 255.0

    ylq_img_arr = [cv2.imread(str(p)) for p in ylq_img_p]
    ylq_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in ylq_img_arr]
    ylq_img = np.array(ylq_img_res).astype(np.float32) / 255.0

    zxz_img_arr = [cv2.imread(str(p)) for p in zxz_img_p]
    zxz_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in zxz_img_arr]
    zxz_img = np.array(zxz_img_res).astype(np.float32) / 255.0

    nh_data = {'thw': {'img': np.array(thw_img), 'labs': [0] * len(thw_img)},
               'ylq': {'img': np.array(ylq_img), 'labs': [2] * len(ylq_img)},
               'zxz': {'img': np.array(zxz_img), 'labs': [3] * len(zxz_img)}}

    return nh_data


def get_ns_data():
    bj_img_p = sort_p(list(Path(r'path/to/ns/bj').glob('*/*.jpg')))
    dwns_img_p = sort_p(list(Path(r'path/to/ns/dwns').glob('*/*.jpg')))
    hsnr_img_p = sort_p(list(Path(r'path/to/ns/hsnr').glob('*/*.jpg')))
    nrz_img_p = sort_p(list(Path(r'path/to/ns/nrz').glob('*/*.jpg')))

    bj_img_arr = [cv2.imread(str(p)) for p in bj_img_p]
    bj_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in bj_img_arr]
    bj_img = np.array(bj_img_res).astype(np.float32) / 255.0

    dwns_img_arr = [cv2.imread(str(p)) for p in dwns_img_p]
    dwns_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in dwns_img_arr]
    dwns_img = np.array(dwns_img_res).astype(np.float32) / 255.0

    hsnr_img_arr = [cv2.imread(str(p)) for p in hsnr_img_p]
    hsnr_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in hsnr_img_arr]
    hsnr_img = np.array(hsnr_img_res).astype(np.float32) / 255.0

    nrz_img_arr = [cv2.imread(str(p)) for p in nrz_img_p]
    nrz_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in nrz_img_arr]
    nrz_img = np.array(nrz_img_res).astype(np.float32) / 255.0

    ns_data = {'bj': {'img': np.array(bj_img), 'labs': [0] * len(bj_img)},
               'dwns': {'img': np.array(dwns_img), 'labs': [1] * len(dwns_img)},
               'hsnr': {'img': np.array(hsnr_img), 'labs': [2] * len(hsnr_img)},
               'nrz': {'img': np.array(nrz_img), 'labs': [3] * len(nrz_img)}}

    return ns_data


def get_xiu_data():
    sux_img_p = sort_p(list(Path(r'path/to/xiu/0/imgs/').glob('*.jpg')))
    xx_img_p = sort_p(list(Path(r'path/to/xiu/1/imgs/').glob('*.jpg')))
    shux_img_p = sort_p(list(Path(r'path/to/xiu/2/imgs/').glob('*.jpg')))
    yx_img_p = sort_p(list(Path(r'path/to/nh/xiu/3/imgs/').glob('*.jpg')))

    sux_img_arr = [cv2.imread(str(p)) for p in sux_img_p]
    sux_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in sux_img_arr]
    sux_img = np.array(sux_img_res).astype(np.float32) / 255.0

    xx_img_arr = [cv2.imread(str(p)) for p in xx_img_p]
    xx_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in xx_img_arr]
    xx_img = np.array(xx_img_res).astype(np.float32) / 255.0

    shux_img_arr = [cv2.imread(str(p)) for p in shux_img_p]
    shux_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in shux_img_arr]
    shux_img = np.array(shux_img_res).astype(np.float32) / 255.0

    yx_img_arr = [cv2.imread(str(p)) for p in yx_img_p]
    yx_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in yx_img_arr]
    yx_img = np.array(yx_img_res).astype(np.float32) / 255.0

    xiu_data = {'sux': {'img': np.array(sux_img), 'labs': [0] * len(sux_img)},
                'xx': {'img': np.array(xx_img), 'labs': [1] * len(xx_img)},
                'shux': {'img': np.array(shux_img), 'labs': [2] * len(shux_img)},
                'yx': {'img': np.array(yx_img), 'labs': [3] * len(yx_img)}}

    return xiu_data

if __name__ == '__main__':

    nh_data = get_nh_data()
    save_pickle(r"nh_data.pickle", nh_data)

    ns_data = get_ns_data()
    save_pickle(r"ns_data.pickle", ns_data)

    xiu_data = get_xiu_data()
    save_pickle(r"xiu_data.pickle", xiu_data)
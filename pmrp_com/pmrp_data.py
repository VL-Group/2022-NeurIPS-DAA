"""
MIT license |
Copyright (c) 2023 Hao Li
"""

import numpy as np
import os
from tqdm import tqdm

def construct_matrix(path1, path2=None):
    if path2:
        matrix = (np.load(path1) + np.load(path2)) / 2
    else:
        matrix = np.load(path1)
    return matrix


def get_dict(matrix, save_dir=None, dataset_root=None, data_split = 'testall'):
    txt_ids, img_ids = [], []

    for line in open(os.path.join(dataset_root, '%s_ids.txt' % data_split), 'rb'):
        img_ids.append(int(line.strip()))

    for line in open(os.path.join(dataset_root, '%s_caption_ids.txt' % data_split), 'rb'):
        txt_ids.append(int(line.strip()))

    img_ids = np.array(img_ids)
    txt_ids = np.array(txt_ids)

    i2t, t2i = {}, {}
    for idx, row in enumerate(tqdm(matrix)):
        key_dict = {}
        retrieved = []
        argsorted = np.argsort(row)[::-1]
        key = img_ids[5 * idx]
        img_argid = img_ids[argsorted]
        txt_argid = txt_ids[argsorted]
        for idx, img_id in enumerate(img_argid):
            id_dict = {}
            id_dict['image_id'] = img_id
            #id_dict['text_id'] = txt_argid[idx]
            retrieved.append(id_dict)

        key_dict['retrieved'] = retrieved
        i2t[key] = key_dict
        
    #np.save('data_i2t.npy', i2t)

    matrix = np.transpose(matrix)
    for idx, row in enumerate(tqdm(matrix)):
        key_dict = {}
        retrieved = []
        argsorted = np.argsort(row)[::-1]
        query_id = txt_ids[idx]
        query_imgid = img_ids[idx]
        argid = img_ids[5 * argsorted]
        for id in argid:
            img_id_dict = {}
            img_id_dict['id'] = id
            retrieved.append(img_id_dict)

        key_dict['query'] = {'image_id': query_imgid}
        key_dict['retrieved'] = retrieved
        t2i[query_id] = key_dict

    #np.save('data_t2i.npy', t2i)

    data = {'i2t': i2t, 't2i': t2i}
    if save_dir:
        save_path = os.path.join(save_dir, 'data.npy')
        np.save(save_path, data)
        print('The data has been saved to {}.'.format(save_path))
        
    return data
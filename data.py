"""Data provider"""

import torch
import torch.utils.data as data
import torch.distributed as dist

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import os
import nltk
import h5py
import numpy as np
import string


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        self.loc = data_path + '/'
        self.img_path = self.loc+'%s.h5' % data_split
        self.data_split = data_split
        self.images = None

        # load the raw captions
        self.captions = []

        for line in open(self.loc+'%s_caps.txt' % data_split, 'rb'):
            self.captions.append(line.strip())

        # load the image features
        #self.images = h5py.File(self.loc+'%s.h5' % data_split, 'r')
        self.length = len(self.captions)

        if data_split == 'train':
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000
            self.captions = self.captions[0:self.length]

        # CIDEr
        self.trans_captions = self.captions

        # TFIDF matrix compute
        self.tfidf_map = self.tfidf_compute(self.captions)
        self.tfidf_shape = (self.tfidf_map.shape[0], self.tfidf_map.shape[1])
        self.idx2GT = np.zeros((self.tfidf_map.shape[0]), dtype=int)
        for i in range(0, self.tfidf_map.shape[0], 5):
            self.idx2GT[i: i + 5] = i



    def tfidf_compute(self, corpus, eps=1e-6):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer()
        tfidf_mat = transformer.fit_transform(X).toarray()

        return tfidf_mat

    def __getitem__(self, index):
        # handle the image redundancy
        if self.images is None:
            self.images = h5py.File(self.loc+'%s.h5' % self.data_split, 'r')
        img_id = index//self.im_div
        image = torch.Tensor(self.images['feat'][img_id])
        caption = self.captions[index]

        # get tfidf vector of indexed caption
        sem_single = self.tfidf_map[index]

        # get tfidf vectors of positive captions of indexed images
        sem_GT = self.tfidf_map[self.idx2GT[index]: self.idx2GT[index] + 5, :]
        #print(type(sem_GT)

        vocab = self.vocab

        # -------- The main difference between python2.7 and python3.6 --------#
        # The suggestion from Hongguang Zhu(https://github.com/KevinLight831)
        # ---------------------------------------------------------------------#
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        # convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, index, img_id, sem_GT, sem_single

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, sem_GT, sem_single = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # CIDEr
    sem_GT = torch.Tensor(np.stack(sem_GT, axis=0))
    sem_single = torch.Tensor(np.stack(sem_single, axis=0))

    return images, targets, lengths, ids, sem_GT, sem_single


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab)
    if data_split == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader, sampler


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader, train_sampler = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader, _ = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    100, False, workers)
    return train_loader, val_loader, train_sampler


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader, test_sampler = get_precomp_loader(dpath, split_name, vocab, opt,
                                     100, False, workers)
    return test_loader


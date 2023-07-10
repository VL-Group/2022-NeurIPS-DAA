"""MS-COCO image-to-caption retrieval dataset code

reference codes:
https://github.com/pytorch/vision/blob/v0.2.2_branch/torchvision/datasets/coco.py
https://github.com/yalesong/pvse/blob/master/data.py
"""

import os
try:
    import ujson as json
except ImportError:
    import json

from PIL import Image
from pycocotools.coco import COCO

from torch.utils.data import Dataset


class CocoCaptionsCap(Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        annFile (string): Path to json annotation file.
        ids (list, optional): list of target caption ids
        extra_annFile (string, optional): Path to extra json annotation file (for training)
        extra_ids (list, optional): list of extra target caption ids (for training)
        instance_annFile (str, optional): Path to instance annotation json (for PMRP computation)
    """
    def __init__(self, annFile, ids=None,
                 extra_annFile=None, extra_ids=None,
                 instance_annFile=None):
        if extra_annFile:
            self.coco = COCO()
            with open(annFile, 'r') as fin1, open(extra_annFile, 'r') as fin2:
                dataset = json.load(fin1)
                extra_dataset = json.load(fin2)
                if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
                    raise TypeError('invalid type {} {}'.format(type(dataset),
                                                                type(extra_dataset)))
                if set(dataset.keys()) != set(extra_dataset.keys()):
                    raise KeyError('key mismatch {} != {}'.format(list(dataset.keys()),
                                                                  list(extra_dataset.keys())))
                for key in ['images', 'annotations']:
                    dataset[key].extend(extra_dataset[key])
            self.coco.dataset = dataset
            self.coco.createIndex()
        else:
            self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            self.ids += list(extra_ids)
        self.ids = [int(id_) for id_ in self.ids]

        self.all_image_ids = set([self.coco.loadAnns(annotation_id)[0]['image_id'] for annotation_id in self.ids])

        iid_to_cls = {}
        if instance_annFile:
            with open(instance_annFile) as fin:
                instance_ann = json.load(fin)
            for ann in instance_ann['annotations']:
                image_id = int(ann['image_id'])
                code = iid_to_cls.get(image_id, [0] * 90)
                code[int(ann['category_id']) - 1] = 1
                iid_to_cls[image_id] = code

            seen_classes = {}
            new_iid_to_cls = {}
            idx = 0
            for k, v in iid_to_cls.items():
                v = ''.join([str(s) for s in v])
                if v in seen_classes:
                    new_iid_to_cls[k] = seen_classes[v]
                else:
                    new_iid_to_cls[k] = idx
                    seen_classes[v] = idx
                    idx += 1
            iid_to_cls = new_iid_to_cls

            if self.all_image_ids - set(iid_to_cls.keys()):
                print(f'Found mismatched! {self.all_image_ids - set(iid_to_cls.keys())}')

        self.iid_to_cls = iid_to_cls
        self.n_images = len(self.all_image_ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        coco = self.coco
        annotation_id = self.ids[index]
        annotation = coco.loadAnns(annotation_id)[0]
        image_id = annotation['image_id']

        return annotation_id, image_id

    def __len__(self):
        return len(self.ids)

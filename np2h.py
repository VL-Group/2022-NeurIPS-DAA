import os
import numpy as np
import h5py


data_path='/home/lihao/data/SGRAF/coco_precomp'   # the path of coco_precomp
print('Converting MS-COCO dataset ...')

for split in ('train', 'dev', 'test', 'testall'):
    Data=np.load(os.path.join(data_path, '{}_ims.npy'.format(split)))
    c=h5py.File(os.path.join(data_path, '{}.h5'.format(split)), 'w')
    c.create_dataset('feat', data=Data)
    c.close()
    print('{} set has been converted!'.format(split))



data_path='/home/lihao/data/SGRAF/f30k_precomp'   # the path of f30k_precomp
print('Converting Flickr30k dataset ...')

for split in ('train', 'dev', 'test'):
    Data=np.load(os.path.join(data_path, '{}_ims.npy'.format(split)))
    f=h5py.File(os.path.join(data_path, '{}.h5'.format(split)), 'w')
    f.create_dataset('feat', data=Data)
    f.close()
    print('{} set has been converted!'.format(split))
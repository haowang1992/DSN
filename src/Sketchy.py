import os,cv2
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import warp, AffineTransform
from nltk.corpus import wordnet as wn
import pickle

def random_transform(img):
    if np.random.random() < 0.5:
        img = img[:,::-1,:]

    if np.random.random() < 0.5:
        sx = np.random.uniform(0.7, 1.3)
        sy = np.random.uniform(0.7, 1.3)
    else:
        sx = 1.0
        sy = 1.0

    if np.random.random() < 0.5:
        rx = np.random.uniform(-30.0*2.0*np.pi/360.0,+30.0*2.0*np.pi/360.0)
    else:
        rx = 0.0

    if np.random.random() < 0.5:
        tx = np.random.uniform(-10,10)
        ty = np.random.uniform(-10,10)
    else:
        tx = 0.0
        ty = 0.0

    aftrans = AffineTransform(scale=(sx, sy), rotation=rx, translation=(tx,ty))
    img_aug = warp(img,aftrans.inverse,preserve_range=True).astype('uint8')

    return img_aug


class SketchyDataset(Dataset):
    def __init__(self, split='train',
                 root_dir='../dataset/Sketchy/',
                 version='sketch_tx_000000000000_ready', zero_version='zeroshot1',
                 cid_mask = False, transform=None, aug=False, shuffle=False,
                 first_n_debug=9999999, contrastive_transform=None):
        
        self.root_dir = root_dir
        self.version = version
        self.split = split
        
        self.img_dir = self.root_dir
        
        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_test.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return
        
        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()
            
        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        if shuffle:
            self.shuffle()
            
        self.file_ls = self.file_ls[:first_n_debug]
        self.labels = self.labels[:first_n_debug]
            
        self.transform = transform
        self.contrastive_transform = contrastive_transform
        self.aug = aug


        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root_dir, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                self.cid_matrix = pickle.load(fh)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.file_ls[idx]))[:,:,::-1]
        if self.aug and np.random.random()<0.7:
            img = random_transform(img)

        if self.contrastive_transform is not None:
            img1 = self.contrastive_transform(img)
            img2 = self.contrastive_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]

        if self.cid_mask:
            mask = self.cid_matrix[label]
            if self.contrastive_transform is not None:
                return img, img1, img2, label, mask
            else:
                return img, label, mask

        else:
            if self.contrastive_transform is not None:
                return img, img1, img2, label
            else:
                return img, label

    
    
    def shuffle(self):
        s_idx = np.random.shuffle(np.arange(len(self.labels)))
        self.file_ls = self.file_ls[s_idx]
        self.labels = self.labels[s_idx]


class SketchImagePairedDataset(Dataset):
    def __init__(self, dataset='sketchy', zero_version='zeroshot1',
                 cid_mask=False, transform=None, contrastive_transform=None,
                 aug=False, shuffle=False, first_n_debug=9999999):
        if dataset == 'sketchy':
            self.root_dir = '../dataset/Sketchy/'
            self.sketch_version = 'sketch_tx_000000000000_ready'
            self.image_version = 'all_photo'
        else:
            self.root_dir = '../dataset/TUBerlin/'
            self.sketch_version = 'png_ready'
            self.image_version = 'ImageResized_ready'

        self.zero_version = zero_version
        self.transform = transform
        self.contrastive_transform = contrastive_transform
        self.aug = aug

        file_ls_sketch = os.path.join(self.root_dir, self.zero_version, self.sketch_version + '_filelist_test.txt')
        file_ls_image = os.path.join(self.root_dir, self.zero_version, self.image_version + '_filelist_train.txt')
        if cid_mask:
            file_dict_cid = os.path.join(self.root_dir, self.zero_version, 'cid_mask.pickle')
            with open(file_dict_semantic, 'rb') as fh:
                self.cid_dict = pickle.load(fh)

        with open(file_ls_sketch, 'r') as fh:
            file_content_sketch = fh.readlines()
        with open(file_ls_image, 'r') as fh:
            file_content_image = fh.readlines()


        self.file_ls_sketch = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content_sketch])
        self.labels_sketch = np.array([int(ff.strip().split()[-1]) for ff in file_content_sketch])
        self.names_sketch = np.array([' '.join(ff.strip().split()[:-1]).split('/')[-2] for ff in file_content_sketch])
        self.file_ls_image = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content_image])
        self.labels_image = np.array([int(ff.strip().split()[-1]) for ff in file_content_image])
        self.names_image = np.array([' '.join(ff.strip().split()[:-1]).split('/')[-2] for ff in file_content_image])

        if shuffle:
            self.shuffle()

        self.file_ls_sketch = self.file_ls_sketch[:first_n_debug]
        self.labels_sketch = self.labels_sketch[:first_n_debug]
        self.names_sketch = self.names_sketch[:first_n_debug]
        self.file_ls_image = self.file_ls_image[:first_n_debug]
        self.labels_image = self.labels_image[:first_n_debug]
        self.names_image = self.names_image[:first_n_debug]

    def __getitem__(self, idx):
        label = self.labels_image[idx]

        select_idx = np.random.choice(np.argwhere(self.labels_sketch == label).reshape(-1), 1)
        sketch = cv2.imread(os.path.join(self.root_dir, self.file_ls_sketch[select_idx][0]))[:, :, ::-1]
        if self.transform is not None:
            sketch = self.transform(sketch)

        image = cv2.imread(os.path.join(self.root_dir, self.file_ls_image[idx]))[:, :, ::-1]
        if self.transform is not None:
            image = self.transform(image)


        return sketch, image, label

    def __len__(self):
        # if self.opt.dataset_name == 'QuickDraw':
        #     return len(self.labels_sketch)
        # else:
        #     return len(self.labels_image)
        return len(self.labels_image)

    def shuffle(self):
        s_idx = np.arange(len(self.labels_sketch))
        np.random.shuffle(s_idx)
        self.file_ls_sketch = self.file_ls_sketch[s_idx]
        self.labels_sketch = self.labels_sketch[s_idx]
        self.names_sketch = self.names_sketch[s_idx]
        s_idx = np.arange(len(self.labels_image))
        np.random.shuffle(s_idx)
        self.file_ls_image = self.file_ls_image[s_idx]
        self.labels_image = self.labels_image[s_idx]
        self.names_image = self.names_image[s_idx]
        
def wnid_to_synset(wnid):
    return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
        

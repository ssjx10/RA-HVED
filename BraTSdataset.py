import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from utils import compute_sdm

def background_info(img, background=0, extract=True, patch_size=112):
    background = img[0,0,0,0]
    brain = np.where(img[0] != background)
    
    min_z = int(np.min(brain[0]))
    max_z = int(np.max(brain[0]))+1
    min_y = int(np.min(brain[1]))
    max_y = int(np.max(brain[1]))+1
    min_x = int(np.min(brain[2]))
    max_x = int(np.max(brain[2]))+1
    
    if max_z - min_z < patch_size:
        pad = patch_size - (max_z - min_z)
        min_pad = pad //2 
        max_pad = pad - min_pad
        add_pad = 0
        min_z = min_z - min_pad
        
        if min_z < 0:
            add_pad -= min_z
            min_z = 0
        
        max_pad += add_pad
        max_z = max_z + max_pad
        
    if max_y - min_y < patch_size:
        pad = patch_size - (max_y - min_y)
        min_pad = pad //2 
        max_pad = pad - min_pad
        add_pad = 0
        min_y = min_y - min_pad
        
        if min_y < 0:
            add_pad -= min_y
            min_y = 0
        
        max_pad += add_pad
        max_y = max_y + max_pad
        
    if max_x - min_x < patch_size:
        pad = patch_size - (max_x - min_x)
        min_pad = pad //2 
        max_pad = pad - min_pad
        add_pad = 0
        min_x = min_x - min_pad
        
        if min_x < 0:
            add_pad -= min_x
            min_x = 0
        
        max_pad += add_pad
        max_x = max_x + max_pad
    
    if extract:
        return min_z, min_y, min_x
    else:
        return 0, 0, 0

def extract_brain(x, background=0, patch_size=112):
    ''' find the boundary of the brain region, return the resized brain image and the index of the boundaries''' 
    img, mask = x
    background = img[0,0,0,0]
    brain = np.where(img[0] != background)
    
    min_z = int(np.min(brain[0]))
    max_z = int(np.max(brain[0]))+1
    min_y = int(np.min(brain[1]))
    max_y = int(np.max(brain[1]))+1
    min_x = int(np.min(brain[2]))
    max_x = int(np.max(brain[2]))+1
    
    if max_z - min_z < patch_size:
        pad = patch_size - (max_z - min_z)
        min_pad = pad //2 
        max_pad = pad - min_pad
        add_pad = 0
        min_z = min_z - min_pad
        
        if min_z < 0:
            add_pad -= min_z
            min_z = 0
        
        max_pad += add_pad
        max_z = max_z + max_pad
        
    if max_y - min_y < patch_size:
        pad = patch_size - (max_y - min_y)
        min_pad = pad //2 
        max_pad = pad - min_pad
        add_pad = 0
        min_y = min_y - min_pad
        
        if min_y < 0:
            add_pad -= min_y
            min_y = 0
        
        max_pad += add_pad
        max_y = max_y + max_pad
        
    if max_x - min_x < patch_size:
        pad = patch_size - (max_x - min_x)
        min_pad = pad //2 
        max_pad = pad - min_pad
        add_pad = 0
        min_x = min_x - min_pad
        
        if min_x < 0:
            add_pad -= min_x
            min_x = 0
        
        max_pad += add_pad
        max_x = max_x + max_pad
            
    
    return img[:, min_z:max_z, min_y:max_y, min_x:max_x], mask[min_z:max_z, min_y:max_y, min_x:max_x]

def normalize(x):
        # x : volume
        p_mean = np.zeros(4)
        p_std = np.zeros(4)
        trans_x = np.transpose(x, (1,2,3,0))
        
        X_normal = (trans_x - np.mean(trans_x[trans_x[:,:,:,0] != 0], 0)) / ((np.std(trans_x[trans_x[:,:,:,0] != 0], 0)) + 1e-6)
        
        # min_max range(-1, 1)
#         cutoff = np.percentile(trans_x, (0, 99))
#         trans_x = np.clip(trans_x, *cutoff)
#         X_normal = (trans_x - np.min(trans_x)) / (np.max(trans_x) - np.min(trans_x))
#         X_normal = X_normal * 2 + -1
        
        return np.transpose(X_normal, (3,0,1,2))

class ISLESset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, indices, transform=None, m_full=False, extract=True, lazy=False):
        'Initialization'
        self.transform = transform
        self.m_full = m_full
        self.tr_idxtoidx = indices # for lazy_load
        self.extract = extract
        self.lazy = lazy
        
        print("data loading...")
        file_path = '/data/isles_siss_2015_3D.hdf5' # (W,H,D)
        with h5py.File(file_path, 'r') as h5_file:
            data = h5py.File(file_path, 'r')
            if lazy:
                self.X = data['images']
                self.mask = data['masks']
            else:
                self.X = data['images'][indices]
                self.mask = data['masks'][indices]               
                

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.tr_idxtoidx)
    
  def load_data(self, idx):
         
        X = self.X[idx]
        mask = self.mask[idx]
        bg_info = background_info(X)
#         bg_info = (0,0,0)
        if self.extract:
            X, mask = extract_brain((X, mask))
        X = normalize(X)
        
        return X, mask, bg_info

  def __getitem__(self, index):
        'Generates one sample of data'
        if self.lazy:
            index = self.tr_idxtoidx[index]
        # Load data and get label
        X, mask, bg_info = self.load_data(index)
        
        if self.transform:
            X, mask = self.transform((X, mask))
        
#         if self.transform:
#             X2, mask2 = self.transform((X, mask))
        missing = X.copy()
        ch1 = np.random.rand()
        ch2 = np.random.rand()
        ch3 = np.random.rand()
        ch4 = np.random.rand()
        modal_check = np.ones(4)  # modal info
        if ch1 > 0.5:
            missing[0] = 0
            modal_check[0] = 0
        if ch2 > 0.5:
            missing[1] = 0
            modal_check[1] = 0
        if ch3 > 0.5:
            missing[2] = 0
            modal_check[2] = 0
        if ch4 > 0.5:
            missing[3] = 0
            modal_check[3] = 0
        
        if min(ch1, ch2, ch3, ch4) > 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = X[chanel_idx]
            modal_check[chanel_idx] = 1
        
        if not self.m_full: # no full modality in missing set
            if max(ch1, ch2, ch3, ch4) < 0.5:
                chanel_idx = np.random.choice(4)
                missing[chanel_idx] = 0
                modal_check[chanel_idx] = 0
                
        return X, missing, mask.astype('float'), bg_info

class GBMset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, indices, transform=None, m_full=False, modal_check=None, full_set=False, extract=True, lazy=False, sdm=False):
        'Initialization'
        self.transform = transform
        self.m_full = m_full
        self.modal_check = modal_check
        self.tr_idxtoidx = indices # for lazy_load
        self.extract = extract
        self.lazy = lazy
        self.sdm = sdm
        self.full_idx = None
        if full_set:
            self.full_idx = np.where(np.sum(modal_check, 1) == 4)[0]
        
        print("data loading...")
        file_path = '/data/brats2018_3D.hdf5'
        with h5py.File(file_path, 'r') as h5_file:
            data = h5py.File(file_path, 'r')
            if lazy:
                self.X = data['images']
                self.mask = data['masks']
            else:
                self.X = data['images'][indices]
                self.mask = data['masks'][indices]               
                

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.tr_idxtoidx)
    
  def load_data(self, idx):
         
        X = self.X[idx]
        mask = self.mask[idx]
        '''for (W,H,D)'''
        X = X.transpose(0,2,1,3) # (H,W,D)240.240,155 -> (W,H,D)240,240,155
        mask = mask.transpose(1,0,2) # (H,W,D)240.240,155 -> (W,H,D)240,240,155
        bg_info = background_info(X)
#         bg_info = (0,0,0)
        if self.extract:
            X, mask = extract_brain((X, mask))
        X = normalize(X)
        
        return X, mask, bg_info

  def __getitem__(self, index):
        'Generates one sample of data'
        if self.modal_check is not None:
            modal_check_orig = self.modal_check[index]
            modal_check = self.modal_check[index].copy()
            for i in range(4):
                if modal_check[i] == 1 and np.sum(modal_check) > 1:
                    modal_check[i] = np.random.randint(2) # random drop
        else:
            modal_check_orig = None
            modal_check = np.random.randint(2, size=(4))
            
        if self.lazy:
            index = self.tr_idxtoidx[index]
        # Load data and get label
        X, mask, bg_info = self.load_data(index)
        
        if self.transform:
            X, mask = self.transform((X, mask))
        if self.sdm:
            sdm_gt = compute_sdm(mask[None])[0]
        
        # fixed missing
        if modal_check_orig is not None:
            if modal_check_orig[0] == 0:
                X[0] = 0
            if modal_check_orig[1] == 0:
                X[1] = 0
            if modal_check_orig[2] == 0:
                X[2] = 0
            if modal_check_orig[3] == 0:
                X[3] = 0
        
        missing = X.copy()
        
        if np.sum(modal_check) == 0:
            chanel_idx = np.random.choice(4)
            modal_check[chanel_idx] = 1
        
        if modal_check[0] == 0:
            missing[0] = 0
        if modal_check[1] == 0:
            missing[1] = 0
        if modal_check[2] == 0:
            missing[2] = 0
        if modal_check[3] == 0:
            missing[3] = 0
        
        if not self.m_full: # no full modality in missing set
            if np.sum(modal_check) == 4:
                chanel_idx = np.random.choice(4)
                missing[chanel_idx] = 0
                modal_check[chanel_idx] = 0
        
        if self.full_idx is not None:
            idx2 = self.full_idx[index % len(self.full_idx)]
            X_full, mask_full, bg_info = self.load_data(idx2)
            X_full, mask_full = self.transform((X_full, mask_full))
            
            return X_full, X, missing, mask_full, mask
                
        if self.sdm:
            return X, missing, (mask, sdm_gt), bg_info
        else:
            return X, missing, mask, bg_info
    
class GBMValidset(Dataset):
  'For brats validationset'
  def __init__(self, extract=True):
        'Initialization'
        self.extract = extract
        
        print("data loading...")
        file_path = '/data/brats2018_3D_validation.hdf5'
        with h5py.File(file_path, 'r') as h5_file:
            data = h5py.File(file_path, 'r')
            self.X = data['images']

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    
  def load_data(self, idx):
        
        X = self.X[idx]
        mask = np.zeros((155, 240, 240)) # dummy
        '''for (W,H,D)'''
        X = np.transpose(X, (0,3,2,1)) # (D,H,W)155,240.240 -> (W,H,D)240,240,155
        mask = np.zeros((240, 240, 155))
        bg_info = background_info(X, extract=self.extract)
#         bg_info = (0,0,0)
        if self.extract:
            X, mask = extract_brain((X, mask))
        X = normalize(X)
        
        return X, bg_info

  def __getitem__(self, index):
        'Generates one sample of data'
        
        # Load data and get label
        X, bg_info = self.load_data(index)
        
        missing = X.copy()
        ch1 = np.random.rand()
        ch2 = np.random.rand()
        ch3 = np.random.rand()
        ch4 = np.random.rand()
        modal_check = np.ones(4)  # modal info
        if ch1 > 0.5:
            missing[0] = 0
            modal_check[0] = 0
        if ch2 > 0.5:
            missing[1] = 0
            modal_check[1] = 0
        if ch3 > 0.5:
            missing[2] = 0
            modal_check[2] = 0
        if ch4 > 0.5:
            missing[3] = 0
            modal_check[3] = 0

        if min(ch1, ch2, ch3, ch4) > 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = X[chanel_idx]
            modal_check[chanel_idx] = 1
            
        if max(ch1, ch2, ch3, ch4) < 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = 0
            modal_check[chanel_idx] = 0

        return X, missing, bg_info
    
class GBMValidset2(Dataset):
  'For brats validationset'
  def __init__(self, extract=True):
        'Initialization'
        
        print("data loading...")
        file_path = '/data/brats2018_3D_validation.hdf5'
        with h5py.File(file_path, 'r') as h5_file:
            data = h5py.File(file_path, 'r')
            X = data['images']
        
        print(X.shape)
#         X = np.transpose(X, (0,1,3,4,2)) # (D,H,W)155,240.240 -> (H,W,D)240,240,155
        X = np.transpose(X, (0,1,4,3,2)) # (D,H,W)155,240.240 -> (W,H,D)240,240,155
        print(X.shape)
        mask = np.zeros((66, 240, 240, 155)) # dummy
        
        print("background info...")
        self.bg_info = [background_info(v, extract=extract) for v in X]
        if extract:
            print("extracting brain...")
            volumes = [extract_brain(v) for v in zip(X, mask)]
        else:
            volumes = zip(X, mask)
        print("normalizing volumes...")
        self.volumes = [(normalize(v), m) for v, m in volumes]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.volumes)

  def __getitem__(self, index):
        'Generates one sample of data'
        
        # Load data and get label
        X, _ = self.volumes[index]
        
        missing = X.copy()
        ch1 = np.random.rand()
        ch2 = np.random.rand()
        ch3 = np.random.rand()
        ch4 = np.random.rand()
        modal_check = np.ones(4)  # modal info
        
        if ch1 > 0.5:
            missing[0] = 0
            modal_check[0] = 0
        if ch2 > 0.5:
            missing[1] = 0
            modal_check[1] = 0
        if ch3 > 0.5:
            missing[2] = 0
            modal_check[2] = 0
        if ch4 > 0.5:
            missing[3] = 0
            modal_check[3] = 0

        if min(ch1, ch2, ch3, ch4) > 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = X[chanel_idx]
            modal_check[chanel_idx] = 1
            
        if max(ch1, ch2, ch3, ch4) < 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = 0
            modal_check[chanel_idx] = 0

        return X, missing, self.bg_info[index]
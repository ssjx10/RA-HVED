import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import glob
from BraTSdataset import GBMset, GBMValidset, GBMValidset2
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import chain, combinations

from unet import UNet3D, ResidualUNet3D, FusionUNet3D, FusionResUNet3D
from RA_HVED import U_HVEDNet3D, U_HVEDConvNet3D
from transform import transforms
from evaluation import eval_overlap_save, eval_overlap, eval_overlap_recon
from utils import seed_everything, all_subsets

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

MODALITIES = [0,1,2,3]
SUBSETS_MODALITIES = all_subsets(MODALITIES)

def parse_args():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument("model_name", help="model name for test")
    parser.add_argument("epoch", type=int, help="model epoch")
    parser.add_argument("--n_class", type=int, default=3, help="number of class")
    parser.add_argument("--save_dir", default='model', help="the dir to save models and logs")
    parser.add_argument("--crop_size", type=int, default=112, help="patch size for inference")
    parser.add_argument("--valid_batch", type=int, default=12, help="batch size for inference")
    parser.add_argument("--d_factor", type=int, default=4, help="stride is crop_size // d_factor ")
    parser.add_argument("--seed", type=int, default=20, help="seed")
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    print('Test', args.model_name, 'epoch', args.epoch)
    seed = args.seed
    crop_size = args.crop_size
    valid_batch = args.valid_batch
    d_factor = args.d_factor
    
    '''dataload'''
    pat_num = 285
    x_p = np.zeros(pat_num,)
    # target value
    y_p = np.zeros(pat_num,)
    indices = np.arange(pat_num)
    x_train_p, x_test_p, y_train_p, y_test_p, idx_train, idx_test = train_test_split(x_p, y_p, indices, test_size=0.2, random_state=seed)
    x_train_p, x_valid_p, y_train_p, y_valid_p, idx_train, idx_valid = train_test_split(x_train_p, y_train_p, idx_train, test_size=1/8, random_state=seed)

    ov_validset = GBMset(sorted(idx_test), transform=transforms(), lazy=False)
    ov_validloader = torch.utils.data.DataLoader(ov_validset, batch_size=1,
                                              shuffle=False, num_workers=4)

    '''model setting'''
    n_class = args.n_class
    model =  U_HVEDConvNet3D(1, n_class,  multi_stream = 4, fusion_level = 4, shared_recon = False,
                    recon_skip=True, MVAE_reduction=True, final_sigmoid=True, f_maps=8, layer_order='ilc')
    model_name = args.model_name
    epoch = args.epoch
    model.load_state_dict(torch.load(f'{args.save_dir}/{model_name}/{epoch}.pth')) 
    model.eval()
    model.cuda()
    
    ''' robust_infer
    T1c T1 T2 F : 14 / T1c T1 T2   : 10 /
    T1c T1      :  4 /     T1      : 1
    '''
    seed_everything(seed)
    tot_eval = np.zeros((2, n_class)) # dice hd95 - wt tc et
    for idx, subset in enumerate(SUBSETS_MODALITIES):
    #     if idx != 14:
    #         continue
        result_text = ''
        if subset[0]:
            result_text += 'T1c '
        else:
            result_text += '    '
        if subset[1]:
            result_text += 'T1 '
        else:
            result_text += '   '
        if subset[2]:
            result_text += 'T2 '
        else:
            result_text += '   '
        if subset[3]:
            result_text += 'FLAIR |'
        else:
            result_text += '      |'
        va_eval = eval_overlap(ov_validloader, model, idx, draw=None, patch_size=crop_size, overlap_stepsize=crop_size//d_factor, batch_size=valid_batch, 
                                  num_classes=n_class, verbose=False, save=False, dir_name=f'{model_name}_{epoch}')
        tot_eval += va_eval
        print(f'{result_text} {va_eval[0][0]*100:.2f} {va_eval[0][1]*100:.2f} {va_eval[0][2]*100:.2f} {va_eval[1][0]:.2f} {va_eval[1][1]:.2f} {va_eval[1][2]:.2f}')
    print(f'{"Average":16s}| {tot_eval[0][0]/15*100:.2f} {tot_eval[0][1]/15*100:.2f} {tot_eval[0][2]/15*100:.2f} {tot_eval[1][0]/15:.2f} {tot_eval[1][1]/15:.2f} {tot_eval[1][2]/15:.2f}')

    
if __name__ == '__main__':
    main()
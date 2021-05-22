import os
import os.path as osp
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchsummary import summary
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from U_HVED import Discriminator, U_HVEDNet3D
from transform import transforms, SegToMask
from loss import DiceLoss, WeightedCrossEntropyLoss, GeneralizedDiceLoss, GANLoss, compute_KLD, compute_KLD_drop
from metrics import MeanIoU, DiceCoefficient, DiceRegion
from evaluation import eval_overlap
from BraTSdataset import GBMset
from utils import subset_idx, seed_everything, init_weights

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"
parallel = False

if __name__ == '__main__':
    
    '''dataload'''
    seed = 20
    seed_everything(seed)
    pat_num = 285
    x_p = np.zeros(pat_num,)
    # target value
    y_p = np.zeros(pat_num,)
    indices = np.arange(pat_num)
    x_train_p, x_test_p, y_train_p, y_test_p, idx_train, idx_test = train_test_split(x_p, y_p, indices, test_size=0.2, random_state=20)
    x_train_p, x_valid_p, y_train_p, y_valid_p, idx_train, idx_valid = train_test_split(x_train_p, y_train_p, idx_train, test_size=1/8, random_state=20)

    train_batch = 3
    crop_size = 112
    valid_batch = 15
    trainset = GBMset(sorted(idx_train), transform=transforms(shift=0.1, flip_prob=0.5, random_crop=crop_size))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    validset = GBMset(sorted(idx_valid), transform=transforms(random_crop=crop_size), m_full=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=valid_batch,
                                              shuffle=False, num_workers=2, pin_memory=True)

    ov_trainset = GBMset(sorted(idx_train), transform=transforms())
    ov_trainloader = torch.utils.data.DataLoader(ov_trainset, batch_size=1,
                                              shuffle=False, num_workers=4)

    ov_validset = GBMset(sorted(idx_valid), transform=transforms())
    ov_validloader = torch.utils.data.DataLoader(ov_validset, batch_size=1,
                                              shuffle=False, num_workers=4)

    ov_testset = GBMset(sorted(idx_test), transform=transforms())
    ov_testloader = torch.utils.data.DataLoader(ov_testset, batch_size=1,
                                              shuffle=False, num_workers=4)
    '''model setting'''
    n_class = 3
    model = U_HVEDNet3D(1, n_class,  multi_stream = 4, fusion_level = 4,
                        recon_skip=True, MVAE_reduction=True, final_sigmoid=True, f_maps=16, layer_order='ilc')
    model.apply(init_weights)
    disc = Discriminator(in_channels=7, ks=4, strides=[1,2,2,2])
    disc.apply(init_weights)
    if parallel:
        model = nn.DataParallel(model)
        disc = nn.DataParallel(disc)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    disc.to(device)

    print_every = 20
    num_epochs= 300
    validate_every = 20
    dir_name = 'model'

    learning_rate = 0.0001
    weight_decay = 0.00001
    alpha = 0.1 # for adv loss
    beta = 0.1 # for recon loss
    train_loss, train_dice = [], []
    valid_loss, valid_dice = [], []

    dice_loss = DiceLoss()
    wce_loss = ce
    gan_loss = GANLoss().to(device)
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    dc = DiceCoefficient()
    dcR = DiceRegion()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_d = optim.Adam(disc.parameters(), lr=0.0001, weight_decay=weight_decay)

    lambda1 = lambda epoch: (1 - epoch / num_epochs)**0.9
    sch = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    sch_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=[lambda1])
    
    for i in range(num_epochs):
        epoch_loss = 0.0
        tr_dice = 0.0
        tr_wt_dice = 0.0
        tr_tc_dice = 0.0
        tr_ec_dice = 0.0

        model.train()
        disc.train()
        start_perf_counter = time.perf_counter()
        start_process_time = time.process_time()
        for x_batch, x_m_batch, mask_batch, _ in trainloader:
            x_batch = x_batch.float().to('cuda')
            x_m_batch = x_m_batch.float().to('cuda')
            mask_batch = mask_batch.float().to('cuda')

            drop = torch.sum(x_m_batch, [2,3,4]) == 0
            subset_size = np.random.choice(range(1,4), 1)
            subset_index_list = subset_idx(subset_size)
            f_outputs, _, f_recon_outputs = model(x_batch, [14], recon=True)
            m_outputs, (mu, logvar), m_recon_outputs = model(x_batch, subset_index_list, recon=True)

            # (1) Update G network about mask + MVAE + GAN
            dice = dice_loss(f_outputs, mask_batch)
            m_dice = dice_loss(m_outputs, mask_batch)
            recon = l2_loss(m_recon_outputs, x_batch)
            sum_prior_KLD = 0.0
            subset_index_list = subset_index_list
            for level in range(len(mu)):
                prior_KLD = compute_KLD(mu[level], logvar[level], subset_index_list)
                sum_prior_KLD += prior_KLD
            KLD = 1/len(mu)*sum_prior_KLD

            syn_f_x = f_recon_outputs.detach()
            syn_m_x = m_recon_outputs
            f_weight = f_outputs.detach()
            f_weight = torch.where(f_weight > 0.5, f_weight, torch.zeros_like(f_weight))
            f_nested_w = f_weight.mean(1)

            m_weight = m_outputs.detach()
            m_weight = torch.where(m_weight > 0.5, m_weight, torch.zeros_like(m_weight))
            m_nested_w = m_weight.mean(1)

            atten_f_x = syn_f_x*(1 + f_nested_w.unsqueeze(1))
            atten_m_x = syn_m_x*(1 + m_nested_w.unsqueeze(1))
            pred_fake = disc(torch.cat([m_outputs, atten_m_x], 1))
            g_gan = gan_loss(pred_fake, True)
            loss = dice + 0.4*m_dice + beta*recon + beta*KLD + alpha*g_gan

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # (2) Update D network
            # train with fake(missing)
            pred_fake = disc(torch.cat([m_outputs.detach(), atten_m_x.detach()], 1))
            loss_d_fake = gan_loss(pred_fake, False)

            # train with real(full)
            pred_real = disc(torch.cat([f_outputs.detach(), atten_f_x.detach()], 1))
            loss_d_real = gan_loss(pred_real, True)

            # Combined D loss
            loss_d = alpha*(loss_d_fake + loss_d_real) * 0.5

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            epoch_loss += loss.item()
            avg_dice = dc(f_outputs.detach(), mask_batch.detach())
            wt_dice = dcR(f_outputs.detach(), mask_batch.detach())
            tc_dice = dcR(f_outputs.detach(), mask_batch.detach(), 'TC')
            ec_dice = dcR(f_outputs.detach(), mask_batch.detach(), 'EC')
            tr_dice += avg_dice.item()
            tr_wt_dice += wt_dice.item()
            tr_tc_dice += tc_dice.item()
            tr_ec_dice += ec_dice.item()

        perf_counter = time.perf_counter() - start_perf_counter
        process_time = time.process_time() - start_process_time
        epoch_loss /= len(trainloader)
        tr_dice /= len(trainloader)
        tr_wt_dice /= len(trainloader)
        tr_tc_dice /= len(trainloader)
        tr_ec_dice /= len(trainloader)

        train_loss.append(epoch_loss)
        train_dice.append(tr_dice)

        va_loss = 0.0
        va_dice = 0.0
        va_wt_dice = 0.0
        va_tc_dice = 0.0
        va_ec_dice = 0.0
        va_wt_dice_m = 0.0
        va_tc_dice_m = 0.0
        va_ec_dice_m = 0.0

        if i<5 or (i + 1) % validate_every == 0:
            with torch.no_grad():
                model.eval()
                disc.eval()
                # Valid accuracy
                for x_batch, x_m_batch, mask_batch, _ in validloader:

                    x_batch = x_batch.float().to('cuda')
                    x_m_batch = x_m_batch.float().to('cuda')
                    mask_batch = mask_batch.long().to('cuda')
                    pred, _ = model(x_batch, valid=True)
                    pred_m, _ = model(x_m_batch, instance_missing=True, valid=True)
                    dice = dice_loss(pred, mask_batch)
                    loss = dice

                    va_loss += loss.item()
                    avg_dice = dc(pred.detach(), mask_batch.detach())
                    wt_dice = dcR(pred.detach(), mask_batch.detach())
                    tc_dice = dcR(pred.detach(), mask_batch.detach(), 'TC')
                    ec_dice = dcR(pred.detach(), mask_batch.detach(), 'EC')
                    wt_dice_m = dcR(pred_m.detach(), mask_batch.detach())
                    tc_dice_m = dcR(pred_m.detach(), mask_batch.detach(), 'TC')
                    ec_dice_m = dcR(pred_m.detach(), mask_batch.detach(), 'EC')

                    va_dice += avg_dice.item()
                    va_wt_dice += wt_dice.item()
                    va_tc_dice += tc_dice.item()
                    va_ec_dice += ec_dice.item()
                    va_wt_dice_m += wt_dice_m.item()
                    va_tc_dice_m += tc_dice_m.item()
                    va_ec_dice_m += ec_dice_m.item()


                va_loss /= len(validloader)
                va_dice /= len(validloader)
                va_wt_dice /= len(validloader)
                va_tc_dice /= len(validloader)
                va_ec_dice /= len(validloader)
                va_wt_dice_m /= len(validloader)
                va_tc_dice_m /= len(validloader)
                va_ec_dice_m /= len(validloader)

                valid_loss.append(va_loss)
                valid_dice.append(va_dice)

        if i == 0:
            print(f'perf_counter per epoch : {time.strftime("%H:%M:%S", time.gmtime(perf_counter))}')
            print(f'process_time per epoch : {time.strftime("%H:%M:%S", time.gmtime(process_time))}')

        if i<5 or (i + 1) % print_every == 0:
            print('Epoch [{}/{}], Train_Loss: {:.4f}, Train_dice: {:.4f}, Train_wt_dice: {:.4f}, Train_tc_dice: {:.4f}, Train_ec_dice: {:.4f},\
                  \nValid_Loss: {:.4f}, Valid_dice: {:.4f}, Valid_wt_dice: {:.4f}, Valid_tc_dice: {:.4f}, Valid_ec_dice: {:.4f},\
                  \nValid_wt_dice: {:.4f}, Valid_tc_dice: {:.4f}, Valid_ec_dice: {:.4f}'
                  .format(i + 1, num_epochs, epoch_loss, tr_dice, tr_wt_dice, tr_tc_dice, tr_ec_dice,
                          va_loss, va_dice, va_wt_dice, va_tc_dice, va_ec_dice,
                         va_wt_dice_m, va_tc_dice_m, va_ec_dice_m))

        if (i + 1) == num_epochs or (i + 1) % 20 == 0:
            print(eval_overlap(ov_testloader, model, patch_size=crop_size, overlap_stepsize=crop_size//2, batch_size=valid_batch, num_classes=3))
        if (i + 1) == num_epochs or (i + 1) % 80 == 0:
            print(eval_overlap(ov_validloader, model, patch_size=crop_size, overlap_stepsize=crop_size//2, batch_size=valid_batch, num_classes=3))
        if (i + 1) == num_epochs or (i + 1) % 80 == 0:
            print(eval_overlap(ov_trainloader, model, patch_size=crop_size, overlap_stepsize=crop_size//2, batch_size=valid_batch, num_classes=3))
        if (i+1) >= 160 and (i + 1) % 10 == 0:
            save_dir = dir_name + '/'
            if parallel:
                m = model.module
            else:
                m = model
            torch.save(m.state_dict(), save_dir + str(i+1) + '.pth')#, _use_new_zipfile_serialization=False)

        sch.step()
        sch_d.step()


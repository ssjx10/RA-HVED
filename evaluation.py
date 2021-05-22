import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from itertools import chain, combinations
from utils import prepare_validation
from metrics import DiceRegion

"""
3D
"""
MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    subsets_modalities = list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))
    return np.array([[True if k in subset else False for k in range(4)] for subset in subsets_modalities])

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def eval_overlap_save(validloader, model, subset_idx=14, draw=None, patch_size=80, overlap_stepsize=8, batch_size=1, num_classes=4, verbose=False, mode='sigmoid'):
    """
        eval_overlap For online evaluation
    """

    
    va_wt_dice = 0.0
    va_tc_dice = 0.0
    va_ec_dice = 0.0
    
    if draw is None: #  mu
        draw = 1
        valid = True
    else: # random draw
        valid = False
        
    cnt = 1
    with torch.no_grad():
            model.eval()
            # Valid accuracy
            preds = []
            for x_batch, _, bg_info in validloader:

                x_batch = x_batch
                
                mod_list = SUBSETS_MODALITIES[subset_idx]
                x_batch[:, mod_list == False] = 0 # drop
                
                _, D, H, W = x_batch[0].shape
                # for bg
                min_x = bg_info[0][0].item()
                min_y = bg_info[1][0].item()
                min_z = bg_info[2][0].item()
                if verbose:
                    print(cnt, (D,H,W), (min_x, min_y, min_z))
                    cnt += 1
                
                drange = list(range(0, D-patch_size+1, overlap_stepsize))
                hrange = list(range(0, H-patch_size+1, overlap_stepsize))
                wrange = list(range(0, W-patch_size+1, overlap_stepsize))

                if (D-patch_size) % overlap_stepsize != 0:
                    drange.append(D-patch_size)
                if (H-patch_size) % overlap_stepsize != 0:
                    hrange.append(H-patch_size)
                if (W-patch_size) % overlap_stepsize != 0:
                    wrange.append(W-patch_size)
                
#                 sum_tot = torch.zeros((1, num_classes, D, H, W))
#                 count_tot = torch.zeros((1, num_classes, D, H, W), dtype=torch.int32)
#                 sum_tot = torch.zeros((1, num_classes, 155, 240, 240))
#                 count_tot = torch.zeros((1, num_classes, 155, 240, 240), dtype=torch.int32)
                sum_tot = torch.zeros((1, num_classes, 240, 240, 155))
                count_tot = torch.zeros((1, num_classes, 240, 240, 155), dtype=torch.int32)
#                 bg_mask = torch.zeros((240, 240, 155), dtype=torch.int32)
#                 bg_mask[min_x:min_x+D, min_y:min_y+H, min_z:min_z+W] = x_batch[0,0]
                patch_ids = []
                base_info = []
                batch_crop = []
                for d in drange:
                    for h in hrange:
                        for w in wrange:
#                             base_info.append([d,h,w])
                            base_info.append([int(bg_info[0])+d,int(bg_info[1])+h,int(bg_info[2])+w])
                            crop_vol = x_batch[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                            batch_crop.append(crop_vol)
#                             crop_mask = mask_batch[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size].long().to('cuda')
                            if len(base_info) == batch_size:
                                base_info = np.stack(base_info)
                                batch_crop = torch.cat(batch_crop).float().to('cuda')
                        
                                pred = 0
                                for k in range(draw):
                                    draw_pred, _ = model(batch_crop, [subset_idx], valid=valid) # crop_image
                                    pred += draw_pred.detach().cpu()
                                pred /= draw
                                
                                for i in range(len(base_info)):
                                    d2 = base_info[i,0]
                                    h2 = base_info[i,1]
                                    w2 = base_info[i,2]
#                                     print(d,h,w)
                                    sum_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += pred[i]
                                    count_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += 1
                            
                                base_info = []
                                batch_crop = []
                                
                if len(base_info) != 0:
                    base_info = np.stack(base_info)
                    batch_crop = torch.cat(batch_crop).float().to('cuda')
                    
                    pred = 0
                    for k in range(draw):
                        draw_pred, _ = model(batch_crop, [subset_idx], valid=valid) # crop_image
                        pred += draw_pred.detach().cpu()
                    pred /= draw

                    for i in range(len(base_info)):
                        d2 = base_info[i,0]
                        h2 = base_info[i,1]
                        w2 = base_info[i,2]
                        sum_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += pred[i]
                        count_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += 1
                        
                results = np.where(count_tot!=0, sum_tot / count_tot, 0)
                results = results[0] # (H,W,D) or (W,H,D)
                
#                 results = results.numpy()
#                 pred = np.zeros((155, 240, 240), dtype=np.int32)
                pred = np.zeros((240, 240, 155), dtype=np.int32)
                if mode == 'softmax':
                    # softmax
                    results = torch.argmax(results, dim=1)
                    
                    idx = np.where(results[1] == 1) # necrotic&NET
                    pred[idx] = 1
                    idx = np.where(results[2] == 1) # edema
                    pred[idx] = 2
                    idx = np.where(results[3] == 1) # ET
                    pred[idx] = 4 # ET
                    
                elif mode == 'sigmoid':
                    # sigmoid
#                     results = np.where(np.max(results, 0) > 0.5, np.argmax(results, 0) + 1, 0) # for high prob
                    results = (results > 0.5)
                    
#                     idx = np.where(results == 1) # WT
                    idx = np.where(results[0] == 1) # WT
                    pred[idx] = 2 # edema
#                     idx = np.where(results == 2) # TC
                    idx = np.where(results[1] == 1) # TC
                    pred[idx] = 1 # necrotic&NET
#                     idx = np.where(results == 3) # ET
                    idx = np.where(results[2] == 1) # ET
                    pred[idx] = 4 # ET
                    
                    # postprocess
                    if np.sum(pred == 4) < 300:
                        pred[pred == 4] = 1
                    
                preds.append(pred)
            preds = np.array(preds)

    return preds

def eval_entire_save(validloader, model, num_classes=4, verbose=False, mode='sigmoid'):
    """
        no patch, segment an entire patient at once.
        v1 : batch 1, extract a brain
    """
    
    va_wt_dice = 0.0
    va_tc_dice = 0.0
    va_ec_dice = 0.0
    
    with torch.no_grad():
            model.eval()
            # Valid accuracy
            preds = []
            for x_batch, _, bg_info in validloader:

                x_batch = x_batch.float().cuda()
                
                
                results = torch.zeros((1, num_classes, 155, 240, 240))

                pred, _ = model(x_batch) # entire_image
                pred = pred.detach().cpu()
                
                d, h, w = bg_info[0][0], bg_info[1][0], bg_info[2][0]
                results[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size] += pred
                
#                 results = results.numpy()
                pred = np.zeros((155, 240, 240), dtype=np.int32)
                if mode == 'softmax':
                    # softmax
                    results = torch.argmax(results, dim=1)
                    
                    idx = np.where(results[1] == 1) # necrotic&NET
                    pred[idx] = 1
                    idx = np.where(results[2] == 1) # edema
                    pred[idx] = 2
                    idx = np.where(results[3] == 1) # ET
                    pred[idx] = 4 # ET
                    
                elif mode == 'sigmoid':
                    # sigmoid
                    results = (results > 0.5)
                    
                    idx = np.where(results[0] == 1) # WT
                    pred[idx] = 2 # edema
                    idx = np.where(results[1] == 1) # TC
                    pred[idx] = 1 # necrotic&NET
                    idx = np.where(results[2] == 1) # ET
                    pred[idx] = 4 # ET
                
                preds.append(pred)
            preds = np.array(preds)

    return preds

def eval_entire_save2(validloader, model, num_classes=4, verbose=False, mode='sigmoid'):
    """
        no patch, segment an entire patient at once.
        v2 : batch n, full volume
    """
    
    va_wt_dice = 0.0
    va_tc_dice = 0.0
    va_ec_dice = 0.0
    
    with torch.no_grad():
            model.eval()
            # Valid accuracy
            results = []
            for x_batch, _, bg_info in validloader:

                x_batch = x_batch.float().cuda()
                
                
                pred, _ = model(x_batch) # entire_image
                result = pred.detach().cpu()
                results.append(result)
                
            results = torch.cat(results, 0)
            preds = np.zeros((66, 155, 240, 240), dtype=np.int32)
            if mode == 'softmax':
                # softmax
                results = torch.argmax(results, dim=1)

                idx = np.where(results[:, 1] == 1) # necrotic&NET
                pred[idx] = 1
                idx = np.where(results[:, 2] == 1) # edema
                pred[idx] = 2
                idx = np.where(results[:, 3] == 1) # ET
                pred[idx] = 4 # ET

            elif mode == 'sigmoid':
                # sigmoid
                results = (results > 0.5)

                idx = np.where(results[:, 0] == 1) # WT
                pred[idx] = 2 # edema
                idx = np.where(results[:, 1] == 1) # TC
                pred[idx] = 1 # necrotic&NET
                idx = np.where(results[:, 2] == 1) # ET
                pred[idx] = 4 # ET
                
            preds = np.array(preds)

    return preds



def eval_overlap(validloader, model, subset_idx=14, draw=None, patch_size=80, overlap_stepsize=8, 
                 batch_size=1, num_classes=4, verbose=False, save=False):
    """
    draw - randomly draw z
    """
    va_wt_dice = 0.0
    va_tc_dice = 0.0
    va_ec_dice = 0.0
    dcR = DiceRegion()
    
    if draw is None: # mu
        draw = 1
        valid = True
    else: # random draw
        valid = False
    
    with torch.no_grad():
            model.eval()
            # Valid accuracy
            cnt = 1
            for x_batch, _, mask_batch, bg_info in validloader:

                x_batch = x_batch
                mask_batch = mask_batch
                
                mod_list = SUBSETS_MODALITIES[subset_idx]
                x_batch[:, mod_list == False] = 0 # drop
                
                _, D, H, W = x_batch[0].shape
                
                drange = list(range(0, D-patch_size+1, overlap_stepsize))
                hrange = list(range(0, H-patch_size+1, overlap_stepsize))
                wrange = list(range(0, W-patch_size+1, overlap_stepsize))

                if (D-patch_size) % overlap_stepsize != 0:
                    drange.append(D-patch_size)
                if (H-patch_size) % overlap_stepsize != 0:
                    hrange.append(H-patch_size)
                if (W-patch_size) % overlap_stepsize != 0:
                    wrange.append(W-patch_size)
                
                sum_tot = torch.zeros((1, num_classes, D, H, W))
                count_tot = torch.zeros((1, num_classes, D, H, W), dtype=torch.int32)
                patch_ids = []
                base_info = []
                batch_crop = []
                for d in drange:
                    for h in hrange:
                        for w in wrange:
                            base_info.append([d,h,w])
                            crop_vol = x_batch[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                            batch_crop.append(crop_vol)
#                             crop_mask = mask_batch[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size].long().to('cuda')
                            if len(base_info) == batch_size:
                                base_info = np.stack(base_info)
                                batch_crop = torch.cat(batch_crop).float().to('cuda')
                                pred = 0
                                
                                for k in range(draw):
                                    draw_pred, _ = model(batch_crop, [subset_idx], valid=valid) # crop_image
                                    pred += draw_pred.detach().cpu()
#                                 pred_soft =  F.softmax(pred.detach().cpu(), dim=1)
                                pred /= draw
                                for i in range(len(base_info)):
                                    d2 = base_info[i,0]
                                    h2 = base_info[i,1]
                                    w2 = base_info[i,2]
#                                     print(d,h,w)
        
                                    sum_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += pred[i]
                                    count_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += 1
                            
                                base_info = []
                                batch_crop = []
                                
                if len(base_info) != 0:
                    base_info = np.stack(base_info)
                    batch_crop = torch.cat(batch_crop).float().to('cuda')
                    pred = 0
                    for k in range(draw):
                        draw_pred, _ = model(batch_crop, [subset_idx], valid=valid) # crop_image
                        pred += draw_pred.detach().cpu()
                    pred /= draw

                    for i in range(len(base_info)):
                        d2 = base_info[i,0]
                        h2 = base_info[i,1]
                        w2 = base_info[i,2]
                        sum_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += pred[i]
                        count_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += 1
                        
                results = sum_tot / count_tot
                wt_dice = dcR(results, mask_batch)
                tc_dice = dcR(results, mask_batch, 'TC')
                ec_dice = dcR(results, mask_batch, 'EC')
                
                va_wt_dice += wt_dice.item()
                va_tc_dice += tc_dice.item()
                va_ec_dice += ec_dice.item()
                if verbose:
                    print(D, H, W)
                    print(wt_dice, tc_dice, ec_dice)
                
                if save:
                    pred = np.zeros((D, H, W), dtype=np.int32)
                    results = (results > 0.5)
                    results = results[0]
                    
                    idx = np.where(results[0] == 1) # WT
                    pred[idx] = 2 # edema
                    idx = np.where(results[1] == 1) # TC
                    pred[idx] = 1 # necrotic&NET
                    idx = np.where(results[2] == 1) # ET
                    pred[idx] = 4 # ET
                    
                    subset = SUBSETS_MODALITIES[subset_idx]
                    subset_name = ''
                    if subset[0]:
                        subset_name += 'T1c'
                    if subset[1]:
                        subset_name += 'T1'
                    if subset[2]:
                        subset_name += 'T2'
                    if subset[3]:
                        subset_name += 'FLAIR'
                    
                    print("Saving " + str(cnt) + " patients...")
                    save_dir = 'robust_result/' + 'tp' + str(cnt) + '/seg/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = save_dir + subset_name + '.nii.gz'   #   get the save path
                    seg = sitk.GetImageFromArray(np.transpose(pred, (2,0,1))) # from (H,W,D) to (D,H,W)
                    # print(seg.GetSize())
                    sitk.WriteImage(seg, save_path)
                    cnt += 1
                
            va_wt_dice /= len(validloader)
            va_tc_dice /= len(validloader)
            va_ec_dice /= len(validloader)

    return va_wt_dice, va_tc_dice, va_ec_dice

def eval_overlap_recon(validloader, model, subset_idx=14, draw=None, patch_size=80, overlap_stepsize=8, 
                       batch_size=1, num_classes=4, verbose=False, save=False):
    """
    draw - randomly draw z
    """
    va_wt_dice = 0.0
    va_tc_dice = 0.0
    va_ec_dice = 0.0
    dcR = DiceRegion()
    
    if draw is None: #  mu
        draw = 1
        valid = True
    else: # random draw
        valid = False
    
    with torch.no_grad():
            model.eval()
            # Valid accuracy
            cnt = 1
            for x_batch, _, mask_batch, bg_info in validloader:
#                 if cnt != 25:
#                     cnt += 1
#                     continue

                x_batch = x_batch
                mask_batch = mask_batch
                
                mod_list = SUBSETS_MODALITIES[subset_idx]
                x_batch[:, mod_list == False] = 0 # drop
                
                _, D, H, W = x_batch[0].shape
                
                drange = list(range(0, D-patch_size+1, overlap_stepsize))
                hrange = list(range(0, H-patch_size+1, overlap_stepsize))
                wrange = list(range(0, W-patch_size+1, overlap_stepsize))

                if (D-patch_size) % overlap_stepsize != 0:
                    drange.append(D-patch_size)
                if (H-patch_size) % overlap_stepsize != 0:
                    hrange.append(H-patch_size)
                if (W-patch_size) % overlap_stepsize != 0:
                    wrange.append(W-patch_size)
                
                sum_tot = torch.zeros((1, num_classes, D, H, W))
                count_tot = torch.zeros((1, num_classes, D, H, W), dtype=torch.int32)
                patch_ids = []
                base_info = []
                batch_crop = []
                for d in drange:
                    for h in hrange:
                        for w in wrange:
                            base_info.append([d,h,w])
                            crop_vol = x_batch[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                            batch_crop.append(crop_vol)
#                             crop_mask = mask_batch[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size].long().to('cuda')
                            if len(base_info) == batch_size:
                                base_info = np.stack(base_info)
                                batch_crop = torch.cat(batch_crop).float().to('cuda')
                                recon_output = 0
                                
                                for k in range(draw):
                                    draw_pred, _, draw_recon = model(batch_crop, [subset_idx], recon=True, valid=valid) # crop_image
                                    recon_output += draw_recon.detach().cpu()
#                                 pred_soft =  F.softmax(pred.detach().cpu(), dim=1)
                                recon_output /= draw
                                for i in range(len(base_info)):
                                    d2 = base_info[i,0]
                                    h2 = base_info[i,1]
                                    w2 = base_info[i,2]
#                                     print(d,h,w)

                                    sum_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += recon_output[i]
                                    count_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += 1
                            
                                base_info = []
                                batch_crop = []
                                
                if len(base_info) != 0:
                    base_info = np.stack(base_info)
                    batch_crop = torch.cat(batch_crop).float().to('cuda')
                    recon_output = 0
                    for k in range(draw):
                        draw_pred, _, draw_recon = model(batch_crop, [subset_idx], recon=True, valid=valid) # crop_image
                        recon_output += draw_recon.detach().cpu()
                    recon_output /= draw

                    for i in range(len(base_info)):
                        d2 = base_info[i,0]
                        h2 = base_info[i,1]
                        w2 = base_info[i,2]
                        sum_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += recon_output[i]
                        count_tot[:, :, d2:d2+patch_size, h2:h2+patch_size, w2:w2+patch_size] += 1
                        
                results = sum_tot / count_tot
                min_val = results.cpu().min()
                print(min_val)
                results[x_batch.cpu() == 0] = min_val - 0.01 
                
                if verbose:
                    print(D, H, W)
                    
                if save:
                    subset = SUBSETS_MODALITIES[subset_idx]
                    subset_name = ''
                    mod_list = ['T1c', 'T1', 'T2', 'FLAIR']
                    if subset[0]:
                        subset_name += mod_list[0]
                    if subset[1]:
                        subset_name += mod_list[1]
                    if subset[2]:
                        subset_name += mod_list[2]
                    if subset[3]:
                        subset_name += mod_list[3]
                        
                    print("Saving " + str(cnt) + " patients...")
                    
                    for i in range(4):
                        save_dir = 'robust_result/' + 'p' + str(cnt) + '/' + mod_list[i] + '/'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_path = save_dir + subset_name + '.nii.gz'   #   get the save path
#                         recon_xi = sitk.GetImageFromArray(np.transpose(results[0, i], (2,0,1))) # H,W,D -> D,H,W
                        recon_xi = sitk.GetImageFromArray(np.transpose(results[0, i], (2,1,0))) # W,H,D -> D,H,W
                        # print(seg.GetSize())
                        sitk.WriteImage(recon_xi, save_path)
                    cnt += 1

    return results

def eval_entire_recon(validloader, model, subset_idx=14, draw=None, num_classes=4, verbose=False, mode='sigmoid', save=False):
    """
        no patch, segment an entire patient at once.
        v1 : batch 1, extract a brain
        v2 : batch n, full volume
    """

    va_wt_dice = 0.0
    va_tc_dice = 0.0
    va_ec_dice = 0.0
    
    if draw is None: #  mu
        draw = 1
        valid = True
    else: # random draw
        valid = False
    
    with torch.no_grad():
            model.eval()
            # Valid accuracy
            cnt = 1
            preds = []
            for x_batch, _, mask_batch, bg_info in validloader:

                x_batch = x_batch.float().cuda()
                
                pred, _, recon = model(x_batch, [subset_idx], recon=True, valid=valid) # entire_image
                recon = recon.detach().cpu()
                
                min_val = recon.cpu().min()
                print(min_val)
#                 recon[x_batch.cpu() == 0] = min_val - 0.01 
                
                if verbose:
                    print(wt_dice, tc_dice, ec_dice)
                    
                if save:
                    subset = SUBSETS_MODALITIES[subset_idx]
                    subset_name = ''
                    mod_list = ['T1c', 'T1', 'T2', 'FLAIR']
                    if subset[0]:
                        subset_name += mod_list[0]
                    if subset[1]:
                        subset_name += mod_list[1]
                    if subset[2]:
                        subset_name += mod_list[2]
                    if subset[3]:
                        subset_name += mod_list[3]
                        
                    print("Saving " + str(cnt) + " patients...")
                    
                    for i in range(4):
                        save_dir = 'robust_result/' + 'p' + str(cnt) + '/' + mod_list[i] + '/'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_path = save_dir + subset_name + '.nii.gz'   #   get the save path
                        recon_xi = sitk.GetImageFromArray(np.transpose(recon[0, i], (2,0,1))) # H,W,D -> D,H,W
#                         recon_xi = sitk.GetImageFromArray(np.transpose(recon[0, i], (2,1,0))) # W,H,D -> D,H,W
                        # print(seg.GetSize())
                        sitk.WriteImage(recon_xi, save_path)
                    cnt += 1

    return recon

def eval_entire(validloader, model, num_classes=4, verbose=False, mode='sigmoid'):
    """
        no patch, segment an entire patient at once.
        v1 : batch 1, extract a brain
        v2 : batch n, full volume
    """

    va_wt_dice = 0.0
    va_tc_dice = 0.0
    va_ec_dice = 0.0
    
    with torch.no_grad():
            model.eval()
            # Valid accuracy
            preds = []
            for x_batch, _, mask_batch, bg_info in validloader:

                x_batch = x_batch.float().cuda()
                
                pred, _ = model(x_batch) # entire_image
                pred = pred.detach().cpu()
                
                wt_dice = dcR(pred, mask_batch)
                tc_dice = dcR(pred, mask_batch, 'TC')
                ec_dice = dcR(pred, mask_batch, 'EC')
                
                va_wt_dice += wt_dice.item()
                va_tc_dice += tc_dice.item()
                va_ec_dice += ec_dice.item()
                if verbose:
                    print(wt_dice, tc_dice, ec_dice)
                
            va_wt_dice /= len(validloader)
            va_tc_dice /= len(validloader)
            va_ec_dice /= len(validloader)

    return va_wt_dice, va_tc_dice, va_ec_dice

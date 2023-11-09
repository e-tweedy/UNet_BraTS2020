import numpy as np
from tqdm.auto import tqdm
import torch
import random
import matplotlib.pyplot as plt
from lib.image_utils import unchunk_image
import pdb

# A useful function for re-seeding all pseudorandom generators
def set_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, dls, optimizer, accelerator, loss_fn, dice, num_training_epochs = 30, encode_mask = True, chunking = True, flipping = True):
    """
    A training loop for segmentation
    Parameters:
    -----------
    model : the UNet model
    dls : tuple(DataLoader,DataLoader)
        the tuple (training dataloader, validation dataloader)
    optimizer : the Pytorch optimizer
    accelerator : the accellerator instance
        Should be prepared prior to passing to this function
    loss_fn : the loss function
    dice : the Dice metric object
    num_training_epochs : int
    encode_mask : bool
        whether masks are encoded as WT, TC, ET
    chunking : bool
        whether to chunk evaluation samples to batch of spatial shape (128,128,128)
        prior to prediction.
        If chunking is False, then will maintain their shape and use test-time evaluation
        methods instead.
    flipping : bool
        if chunking, whether to randomly flip spatial dimensions before predicting
        if chunking is False, this will be ignored
        
    Returns:
    --------
    model : the trained UNet model
    loss_hist_train : the log of training loss
    loss_hist_val : the log of validation loss
    metrics_hist_val : the log of metric scores on validation set
    
    """
    train_dl, valid_dl = dls
    
    num_update_steps_per_epoch = len(train_dl)
    num_training_steps = num_training_epochs*num_update_steps_per_epoch
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    progress_bar = tqdm(range(num_training_steps))
    loss_hist_train = [0]*(num_training_epochs)
    loss_hist_val = [0]*(num_training_epochs)
    metrics_hist_val = {}
    # for prefix in ['iou_','dice_']:
    for prefix in ['dice_']:
        if encode_mask:
            for suffix in ['avg','et','tc','wt']:
                metrics_hist_val[prefix+suffix]=[0]*(num_training_epochs)
        else:
            metrics_hist_val[prefix+'avg']=[0]*(num_training_epochs)
            for i in range(4):
                metrics_hist_val[prefix+str(i)]=[0]*(num_training_epochs)

    set_seed()
    for epoch in range(num_training_epochs):
        model.train()
        # print(f'Epoch: {epoch}, LR: {lr_scheduler.get_last_lr()}')
        for step, (image,mask) in enumerate(train_dl):
            outputs = model(image)
            loss = loss_fn(outputs,mask)
            accelerator.backward(loss)
        
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            loss_hist_train[epoch] += loss.item()*mask.size(0)
    
        loss_hist_train[epoch] /= len(train_dl.dataset)
    
        #if (num_training_epochs - epoch <= 5)|(epoch % 5 == 0):
        if epoch >= 0:
            model.eval()
            with torch.no_grad():
                accelerator.print('Evaluating:')
                for sample in tqdm(valid_dl,leave=False):
                    if chunking:
                        image,mask,extra_data,path = sample
                        if flipping: do_flips, overlaps = extra_data
                        else: overlaps = extra_data
                        image = image[0]
                        outputs = model(image)
                        if flipping:
                            for idx in range(outputs.shape[0]):
                                for axis in range(len(do_flips[idx])):
                                    unflip = torch.flip(outputs[idx],dims = (axis+1,)) if do_flips[idx][axis] else outputs[idx]
                                    outputs[idx] = unflip
                        outputs = unchunk_image(outputs, overlaps)
                        loss = loss_fn(outputs,mask)
                        loss_hist_val[epoch] += loss.item()
                  
                    else:
                    # image has shape (B,A,C,W,H,D) where B is batch size,
                    # A is number of TTA duplicates, C is number of channels
                    # mask has shape (B,3,W,H,D) if encoded or (B,W,H,D) if not
                        image,mask,do_flips,path = sample
                        
                        # swap the augmentation index to the front
                        image = torch.transpose(image, 0,1)
                        num_augments = image.shape[0]
            
                        # These will have shape (B,C,W,H,D) for outputs
                        # and (B,C,W,H,D) or (B,W,H,D) for mask
                        in_shape = list(image[0].shape)
                        outputs = torch.zeros(in_shape).to(device)
                        for aug_idx in range(num_augments):
                            # image_aug has shape (B, C,W,H,D) and mask_aug
                            # has shape (B,C,W,H,D) or (B,W,H,D)
                            image_aug = image[aug_idx]
                            outputs_aug = model(image_aug)
                            for axis in range(len(image_aug.shape)-2):
                                outputs_aug = torch.flip(outputs_aug,dims = (axis+2,)) if do_flips[aug_idx][axis] else outputs_aug
                            outputs += outputs_aug
            
                        outputs /= num_augments
                        loss = loss_fn(outputs,mask)
                        loss_hist_val[epoch] += loss.item()*image.size(0)
            
                    # batch_acc = acc(outputs,mask)
                    # batch_iou = iou(outputs,mask)
                    batch_dice = dice(outputs,mask)
            

                    for i,suffix in enumerate(['et','tc','wt']):
                        # metrics_hist_val['iou_'+suffix][epoch] += batch_iou[i]
                        metrics_hist_val['dice_'+suffix][epoch] += batch_dice[i]

                    # metrics_hist_val['iou_avg'][epoch] += torch.mean(batch_iou)
                    metrics_hist_val['dice_avg'][epoch] += torch.mean(batch_dice)
            
                for key in metrics_hist_val:
                    metrics_hist_val[key][epoch] /= len(valid_dl)
                if chunking: num_augments=1
                loss_hist_val[epoch] /= num_augments*len(valid_dl.dataset)

            print(f'Epoch {epoch+1} ----- training loss: {loss_hist_train[epoch]:.4f} ----- validation loss: {loss_hist_val[epoch]:.4f} ')    
            # print(f'         ----- Avg. scores ... IoU: {metrics_hist_val["iou_avg"][epoch]:.4f} ... Dice: {metrics_hist_val["dice_avg"][epoch]:.4f} ')
            # print(f'         ----- ET. scores ... IoU: {metrics_hist_val["iou_et"][epoch]:.4f} ... Dice: {metrics_hist_val["dice_et"][epoch]:.4f} ')
            # print(f'         ----- TC. scores ... IoU: {metrics_hist_val["iou_tc"][epoch]:.4f} ... Dice: {metrics_hist_val["dice_tc"][epoch]:.4f} ')
            # print(f'         ----- WT. scores ... IoU: {metrics_hist_val["iou_wt"][epoch]:.4f} ... Dice: {metrics_hist_val["dice_wt"][epoch]:.4f} ')
            print(f'         ----- Avg  Dice: {metrics_hist_val["dice_avg"][epoch]:.4f} ')
            print(f'         ----- ET   Dice: {metrics_hist_val["dice_et"][epoch]:.4f} ')
            print(f'         ----- TC   Dice: {metrics_hist_val["dice_tc"][epoch]:.4f} ')
            print(f'         ----- WT   Dice: {metrics_hist_val["dice_wt"][epoch]:.4f} ')
            print('==========')
    return model, loss_hist_train, loss_hist_val, metrics_hist_val

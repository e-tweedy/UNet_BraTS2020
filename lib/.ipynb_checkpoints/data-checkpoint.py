import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from lib.image_utils import *

class MRIDataset(Dataset):
    """
    Custom torch.utils.data.Dataset class for MRI data
    Parameters:
    -----------
    paths : list(str)
        List of paths prefixes for files
    training : bool
        Whether this is a training dataset (as opposed to validation or testing)
    dupe_factor : int
        How many augmentation samples to produce for each training sample if training, or else
        How many test-time-augmentation samples to produce for each validation/testing sample
    noise_prob : float
        probability that an augmented sample received random Gaussian noise
    """
    def __init__(self,paths, training = True, dupe_factor = 4, noise_prob = 0.5, flipping = True,chunking = True, random_seed = 42):
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.paths = paths
        self.training = training
        self.dupe_factor = dupe_factor
        self.noise_prob = noise_prob
        self.random_seed = random_seed
        self.chunking = chunking
        self.flipping = flipping
        
    def __len__(self):
        """
        self.train_dupe_factor dataset items are produced
        for each training set sample via data augmentation
        Augmentations are stacked for validation/testing samples,
        and so only one dataset item is produced per sample
        """
        if self.training:
            return self.dupe_factor*len(self.paths)
        else:
            return len(self.paths)

    def __getitem__(self,
                    index:int):
        # If this is a training dataset, the dataset item at each index
        # will be an augmentation of the sample whose position in self.paths
        # is equal to the residue of index modulo len(self.paths)
        if self.training:
            index = index % len(self.paths)

        # Get stored image and mask
        image_path = 'images/'+self.paths[index]
        mask_filename = self.paths[index].replace('image','mask')
        mask_path = 'masks/'+mask_filename
       
        image = np.load(image_path)
        mask = np.load(mask_path)
        
        if self.training:
            # Random crop to (128,128,128) or smaller
            # random.seed(self.random_seed)
            image, mask = crop_to_size(image,mask,method='random')
            
            # Pad out to (128,128,128) as necessary
            image, mask = pad_to_size(image,mask,target_shape=(128,128,128))

            # Add Gaussian noise with probability self.noise_prob
            # Noise has mean 0 and variance equal to 0.1
            # times the variance of the image voxel intensity
            image = randomly_add_noise(image,prob = self.noise_prob)
            
            # For each spatial axis, perform a flip
            # with probability 0.5
            image,mask,_ = random_flips(image,mask)
        
        else:
            if self.chunking:
                im_shape = image.shape[1:]
                if min(im_shape)<128:
                    im_shape = [max(l,128) for l in im_shape]
                    image, mask = pad_to_size(image,mask,target_shape = im_shape)
                image, overlaps = chunk_image(image)
                if self.flipping:
                    do_flips = []
                    for idx in range(image.shape[0]):
                        flipped_chunk, do_flips_chunk = random_flips(image[idx])
                        image[idx] = flipped_chunk
                        do_flips.append(do_flips_chunk)
            
            else:
                image_augs, mask_augs, do_flips = [],[],[]
                for i in range(self.dupe_factor):
                    # Randomly add Gaussian noise and flips
                    image_aug = randomly_add_noise(image,prob = self.noise_prob)
                    image_aug, mask_aug, do_flips_aug = random_flips(image_aug, mask)
                    image_augs.append(image_aug)
                    mask_augs.append(mask_aug)
                    do_flips.append(do_flips_aug)
            
                # Stack augmentations into one image of shape
                # (self.dupe_factor,3,H,W,D)
                image = np.stack(image_augs,axis=0)
        
        image = torch.from_numpy(image).type(self.inputs_dtype)
        mask = torch.from_numpy(mask).type(self.targets_dtype)
        if self.training:
            return image, mask
        
        # If validation or testing set, we need the do_flips/chunking overlap information
        # in order to un-flip/unchunk predictions
        else:
            if self.chunking:
                if self.flipping: return image, mask, (do_flips, overlaps), self.paths[index]
                else: return image, mask, overlaps, self.paths[index]
            else:
                return image, mask, do_flips, self.paths[index]
        
def get_dl(paths, training = True, batch_size = 1, dupe_factor = 1, noise_prob = 0.8, chunking = True, flipping = True):
    ds = MRIDataset(
        paths, training, dupe_factor = dupe_factor, noise_prob = noise_prob, chunking = chunking, flipping = flipping,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True
    )
    return dl
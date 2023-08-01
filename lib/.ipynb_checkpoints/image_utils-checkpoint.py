import torch
from torch.utils.data import Dataset
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from skimage.transform import rescale
import random
from skimage.util import random_noise

def get_mask(prefix):
    """
    Load segmentation mask from .nii file, change 4->3 so labels are 0,1,2,3
    Parameters:
    -----------
    prefix : str
        the path to the filename, but missing 'seg.nii'
    Returns:
    --------
    mask : numpy.array
        the segmentation mask of shape (H,W,D)
    """
    mask = nib.load(prefix+'seg.nii').get_fdata().astype(np.uint8)
    # mask = pad_crop_resize(mask,preserve_range=True).astype(np.uint8)
    mask[mask==4]=3
    return mask     

def get_stacked_image(prefix):
    """
    Load stacked 3-channel (T1-CE, T2, FLAIR) image with channels normalized
    Parameters:
    -----------
    prefix : str
        the path to the filename, but missing '<type>.nii'
    Returns:
    --------
    image : numpy.array
        the image of shape (3,H,W,D)
    """
    scaler = MinMaxScaler()
    # Load images and rescale pixel values to interval [0,1]
    flair_img = nib.load(prefix+'flair.nii').get_fdata()
    flair_img = scaler.fit_transform(flair_img.reshape(-1,flair_img.shape[-1])).reshape(flair_img.shape)
        
    t1ce_img = nib.load(prefix+'t1ce.nii').get_fdata()
    t1ce_img = scaler.fit_transform(t1ce_img.reshape(-1,t1ce_img.shape[-1])).reshape(t1ce_img.shape)
    
    t2_img = nib.load(prefix+'t2.nii').get_fdata()
    t2_img = scaler.fit_transform(t2_img.reshape(-1,t1ce_img.shape[-1])).reshape(t2_img.shape)

    # Stack three image types as channels
    image = np.stack([flair_img,t1ce_img,t2_img],axis=0)
    
    return image
    
def get_crop_slice(target_size,width,method = 'symmetric'):
    """
    Generate slice object for cropping image axis to target width
    Parameters:
    -----------
    target_length : int
        the desired length of the cropped axis
        if target_length > length, will return slice(0,length)
    length : int
        the current length of the axis
    method : str
        cropping method - must be 'symmetric' or 'random'
        if symmetric, will crop roughly half from each end
        if random, will choose a random window of length target_length
    Returns:
    --------
    the slice object
    """
    assert method in ['random','symmetric'], "method must be 'random' or 'symmetric'"
    if width > target_size:
        crop_extent = width - target_size
        if method == 'random':
            start = random.randint(0,crop_extent)
        else:
            start = crop_extent//2
        stop = start+target_size
        return slice(start,stop)
    else:
        return slice(0,width)

def get_pad_widths(width,target_width = None):
    """
    Generate tuple of padding widths for padding an axis
     Will pad roughly evenly on either end of the axis
    Parameters:
    -----------
    target_length : int of None
        the desired length of the padded axis
        if target_length < length, will return (0,0)
        if target_length is None, will pad the length
        upto the next multiple of 16
    length : int
        the current length of the axis
    Returns:
    --------
    (left,right) : (int,int)
        the number of pixels on each end of the axis
    """
    if target_width is None:
        if width % 16 == 0:
            return (0,0)
        else:
            target_width = 16*(width//16+1)
    
    if width >= target_width:
            return (0,0)
    else:
        pad_extent = target_width - width
        left = pad_extent//2
        right = pad_extent - left
        return (left,right)

def randomly_add_noise(image, fraction = 0.1, prob = 0.5):
    """
    Add random Gaussian noise to an image with some probability
    Gaussian has mean 0 and variance equal to fraction times the
    variance of the image pixel intensity
    Parameters:
    -----------
    image : numpy.array
        the input image to be noised
    fraction : float
        the fraction (desired noise variance) / (image intensity variance)
    prob : float
        the probability that noise is added
    Returns:
    --------
    image : numpy.array
        the (possibly) noised image
    """
    rand_num = random.random()
    if rand_num < prob:
        image = random_noise(image, var = fraction*np.std(image)**2)
    return image

def flip_axes(image, do_flips, mask=None):
    for i in range(len(do_flips)):
        image = np.flip(image,axis=i+1) if do_flips[i] else image
        if mask is not None:
            mask_adjust = 1 if len(mask.shape)==len(image.shape) else 0
            mask = np.flip(mask,axis=i+mask_adjust) if do_flips[i] else mask
    if mask is not None:
        return image.copy(), mask.copy()
    else:
        return image.copy()
            


def random_flips(image,mask=None):
    """
    Flip each axis of an image randomly, where each axis
    has probability 0.5 of being flipped
    Parameters:
    -----------
    image : numpy.array
        the input image to be flipped
    mask : numpy.array or None
        the mask to be flipped (optional)
    Returns:
    --------
    image : numpy.array
        the (possibly) flipped image
    mask : numpy.array (if mask provided)
        the (possibly) flipped mask
    do_flips : list(bool)
        the record of whether flips were done for each axis
        (useful for unflipping if necessary)
    """
    do_flips = []
    for axis in range(1,len(image.shape)):
        do_flips.append(random.choice([True,False]))
    if mask is not None:
        image, mask = flip_axes(image,do_flips,mask)
        return image, mask, do_flips
    else:
        image = flip_axes(image,do_flips)
        return image, do_flips
        # image = np.flip(image,axis=axis) if do_flip[-1] else image
        # if mask is not None:
        #     mask_adjust = 0 if len(mask.shape)==len(image.shape) else -1
        #     mask = np.flip(mask,axis=axis+mask_adjust) if do_flip[-1] else mask
    # if mask is not None:
    #     return image.copy(), mask.copy(), do_flips
    # return image.copy(), do_flips
    
def crop_border(image, mask = None):
    """
    Crop each axis of the image (and optionally a mask)
    so that it has only 1 pixel of of 0's on each side of every axis.
    Parameters:
    -----------
    image : numpy.array
        the input image to be cropped
    mask : numpy.array or None
        the optional mask to also be cropped
    Returns:
    --------
    image, mask (if provided) : numpy.array
        the cropped image and mask (if provided)
    """
    # Cut off border of zeros (leave one pixel margin on each axis)
    xind,yind,zind = np.nonzero(np.sum(image,axis=0) > 1e-5)
    xmin,ymin,zmin = [max(0,np.min(arr)-1) for arr in (xind,yind,zind)]
    xmax,ymax,zmax = [np.max(arr)+1 for arr in (xind,yind,zind)]
    image = image[:,xmin:xmax,ymin:ymax,zmin:zmax]
    if mask is not None:
        if len(mask.shape) == 4:
            mask = mask[:,xmin:xmax,ymin:ymax,zmin:zmax]
        else:
            mask = mask[xmin:xmax,ymin:ymax,zmin:zmax]
        return image, mask
    return image

def crop_to_size(image,mask=None,target_shape=(128,128,128),method='symmetric'):
    """
    Crop the image (and optionally a mask)
    to a target shape
    Parameters:
    -----------
    image : numpy.array
        the input image to be cropped
    mask : numpy.array or None
        the optional mask to also be cropped
    target_shape : tuple(int)
        the desired shape of the spatial axes
    method : str
        cropping method - must be 'symmetric' or 'random'
        if symmetric, will crop roughly half from each end of each axis
        if random, will choose a random window of length target_length for each axis
    Returns:
    --------
    image, mask (if provided) : numpy.array
        the cropped image and mask (if provided)
    """
    assert method in ['random','symmetric'], "method must be 'random' or 'symmetric'"
    slices = [get_crop_slice(target_shape[i],width,method=method)\
              for i,width in enumerate(image.shape[1:])]
    image = image[:,slices[0],slices[1],slices[2]]
    if mask is not None:
        if len(mask.shape) == 4:
            mask = mask[:,slices[0],slices[1],slices[2]]
        else:
            mask = mask[slices[0],slices[1],slices[2]]
        return image, mask
    return image
    
def pad_to_size(image,mask=None,target_shape=None):
    """
    Pad the image (and optionally a mask)
    to a target shape if provided, or so that each
    axis is a multiple of 16
    Padding will be roughly half on each end of each axis
    Parameters:
    -----------
    image : numpy.array
        the input image to be padded
    mask : numpy.array or None
        the optional mask to also be padded
    target_shape : tuple(int)
        the desired shape of the spatial axes
    Returns:
    --------
    image, mask (if provided) : numpy.array
        the padded image and mask (if provided)
    """
    if target_shape is None:
        pads = [get_pad_widths(width) for width in image.shape[1:]]
    else:
        pads = [get_pad_widths(width,target_shape[i]) for i,width in enumerate(image.shape[1:])]
    image = np.pad(image,[(0,0)]+pads)
    if mask is not None:
        if len(mask.shape) == 4:
            mask = np.pad(mask,[(0,0)]+pads)
        else:
            mask = np.pad(mask,pads)
        return image, mask
    return image

def encode_mask(mask):
    """
    Encode a segmentation mask using one-hot-encoding
    based on the classes ET, TC, WT
    Encoding: 0 -> none, 1-> TC and WT; 2-> WT; 3 -> TC and WT and ET
    Parameters:
    -----------
    mask : numpy.array
        the mask to be encoded.  Shape should be (H,W,D)
        or (B,H,W,D) if batched
        and entries are ints from 0,1,2,3
    Returns:
    --------
    mask : numpy.array
        the one-hot encoded mask of shape (3,H,W,D)
        or (B,3,H,W,D) if batched
    """
    assert len(mask.shape) in [4,3], 'Non-encoded mask should have shape (H,W,D) or (B,H,W,D)'
    channel_axis = 1 if len(mask.shape)==4 else 0
    et = mask==3
    tc = np.logical_or(mask==3,mask==1)
    wt = mask > 0
    return np.stack([et,tc,wt],axis=channel_axis).astype(np.int8)

def decode_mask(mask):
    """
    Decode an encoded segmentation mask
    Decoding: [0,0,0] -> 0; [0,1,1] -> 1; [0,0,1] -> 2; [1,1,1] -> 3
    Parameters:
    -----------
    mask : numpy.array
        the mask to be decoded.  Shape should be (3,H,W,D) or (B,3,H,W,D)
        if batched, and every slice (3,i,j,k) should be either
        [0,0,0] (none), [0,1,1] (TC and WT), [0,0,1] (WT), or [1,1,1] (TC,WT,ET)
    Returns:
    --------
    mask : numpy.array
        the decoded mask of shape (H,W,D) or (B,H,W,D) if batched
        with entries from 0,1,2,3
    """
    assert len(mask.shape) in [4,5], 'Encoded mask should have shape (3,H,W,D) or (B,3,H,W,D)'
    channel_axis = 1 if len(mask.shape)==5 else 0
    [et,tc,wt] = np.split(mask,mask.shape[channel_axis],axis=channel_axis)
    et = np.squeeze(et)
    tc = np.squeeze(tc)
    wt = np.squeeze(wt)
    mask = np.zeros(shape = et.shape,dtype=np.uint8)
    mask[et==1] = 3
    mask[np.logical_and(tc==1,et==0)] = 1
    mask[np.logical_and(wt==1,tc==0)] = 2
    return mask
    
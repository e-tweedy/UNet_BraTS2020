import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import random

def plot_sample(image,mask,nslice = None,pred_mask = None, im_cmap = 'gray', seg_cmap = 'viridis'):
    """
    Plots four images at the z-value specified
    by nslice: (1) T1CE image, (2) T2 image,
    (3) FLAIR image, and (4) segmentation mask
    Parameters:
    -----------
    image : numpy.array
        An image array of shape (3,W,H,D)
    mask : numpy.array
        A segmentation mask array of shape (W,H,D)
    pred_mask : numpy.array or None
        Segmentation mask predicted by our model,
        of shape (W,H,D).  If None, will not plot this
    nslice : int or None
        The slice along the z-axis to plot
        If None, then will choose a random slice,
        avoiding the 30-pixel region near either edge
    im_cmap : str
        The colormap to use in the image plots
    seg_cmap : str
        The colormap to use in the segmentation plots
    """
    if nslice is None:
        nslice = random.randint(30,image.shape[-1]-30)
    print(f'Slice {nslice} out of {mask.shape[2]}:')
    
    num_plots = 4 if pred_mask is None else 5
    fig, axs = plt.subplots(1,num_plots,figsize=(num_plots*3,4))
    for i in range(3):
        axs[i].imshow(image[i,:,:,nslice],cmap=im_cmap)
    axs[3].imshow(mask[:,:,nslice],cmap=seg_cmap)
    axs[0].set_title('T1CE image')
    axs[1].set_title('T2 image')
    axs[2].set_title('FLAIR image')
    axs[3].set_title('Segmentation mask')
    if pred_mask is not None:
        axs[4].imshow(pred_mask[:,:,nslice],cmap=seg_cmap)
        axs[4].set_title('Segmentation mask - predicted')
    norm = mpl.colors.Normalize(vmin=0,vmax=3)
    sm = plt.cm.ScalarMappable(cmap=seg_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0,3,4),orientation='horizontal')
    plt.show()
    
def animate_sample(image, mask, pred_mask = None, im_cmap = 'gray', seg_cmap = 'viridis'):
    """
    Animates four images by varying along
    the z-axis: (1) T1CE image, (2) T2 image,
    (3) FLAIR image, and (4) segmentation mask
    Parameters:
    -----------
    image : numpy.array
        An image array of shape (3,W,H,D)
    mask : numpy.array
        A segmentation mask array of shape (W,H,D)
    pred_mask : numpy.array or None
        Segmentation mask predicted by our model,
        of shape (W,H,D).  If None, will not plot this
    cmap : str
        The colormap to use in the plot
    """
    num_plots = 4 if pred_mask is None else 5
    fig, axs = plt.subplots(1,num_plots,figsize=(num_plots*3,4))
    frames = []
    for i in range(0,mask.shape[2]):
        frame = []
        for j in range(3):
            frame.append(axs[j].imshow(image[j,:,:,i],cmap=im_cmap))
        frame.append(axs[3].imshow(mask[:,:,i],cmap=seg_cmap, vmin=0, vmax=3))
        if pred_mask is not None:
            frame.append(axs[4].imshow(pred_mask[:,:,i],cmap=seg_cmap, vmin=0, vmax=3))
        frames.append(frame)
    axs[0].set_title('T1CE image')
    axs[1].set_title('T2 image')
    axs[2].set_title('FLAIR image')
    axs[3].set_title('Segmentation mask')
    if pred_mask is not None:
        axs[4].set_title('Segmentation mask - predicted')
    ani = animation.ArtistAnimation(fig,frames,interval=50,blit=True)
    norm = mpl.colors.Normalize(vmin=0,vmax=3)
    sm = plt.cm.ScalarMappable(cmap=seg_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0,3,4),orientation='horizontal')
    plt.close()
    return ani

def plot_training_curve(prefix = None, num_training_epochs, train_loss = None, val_loss = None, save_fig = False):
    """
    Retrieve loss logs and plot the training curve
    Parameters:
    -----------
    prefix : str
        The model file prefix.  If train_loss and val_loss are not provided,
        will use this prefix to retrieve those logs from model results pkl file
    num_training_epochs : int
        The number of epochs the model was trained for
    save_fig : bool
        Whether to save the figure
    train_loss : list
        Training loss values, optional
    val_loss : list
        Validation loss values, optional
    """
    if train_loss is None or val_loss is None:
        with open(prefix+'_results.pkl','rb') as f:
            _, train_loss, val_loss = pickle.load(f)
    plt.plot(range(num_training_epochs),train_loss, color='b', label='Training loss')
    plt.plot(range(num_training_epochs),val_loss, color='r', label='Validation loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Training curves over {num_training_epochs} epochs")
    plt.legend()
    if save_fig:
        plt.savefig(prefix+'_training_curve.png')
    plt.show()
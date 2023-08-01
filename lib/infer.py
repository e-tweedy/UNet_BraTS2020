import torch
from tqdm.auto import tqdm
import pickle

def evaluate_loop(model, dl, metrics, metric_names, return_sample_scores = False, save_extremes=False):
    """
    A model evaluation loop
    
    Parameters:
    -----------
    model : torch.nn.Module
        the model to use for prediction
    dl : torch.data.utils.DataLoader
        the dataloader of samples to evaluate.  Generates tuples
        (image, mask, do_flips) where:
        - image : torch.tensor
            the batched and augmented image tensor
            shape (B, A, C, H, W, D)
        - mask : torch.tensor
            the batched segmentation mask
            shape (B, C, H, W, D) or (B, H, W, D)
            depending on whether masks are encoded
        - do_flips : list(list(bool))
            list of flip instructions for each image augmentation
            do_flips[i][j] is whether axis j on augmentation i should be flipped
    metrics : list(torchmetrics metric)
        the metrics to use for evaluation
    metric_names : list(str)
        the names of the metrics as strings (for dictionary keys)
    return_sample_scores : bool
        whether to return individual sample scores
    save_extremes : bool
        whether to save outputs from samples with extreme average dice scores
        
    Returns:
    --------
    scores : dict
        dictionary of average metric scores among all samples in dl
        keys are '<name>_<class>' for
        name in metric_names and for class in 'et','tc','wt'
    sample_scores : list(tuple(str,dict)), only returned if return_sample_scores is True
        sample_scores[i] is a tupel (path,scores) where path is the filename of the
        sample as a string and scores is the dictionary of scores for that string,
        formatted as described above
    
    """
    sample_scores = []
    for path,image,mask,do_flips in tqdm(dl,leave=False):
        path = path[0]
        outputs = tta_predict(model, image, do_flips)
        sample_score = score(outputs,mask,metrics,metric_names)
        sample_scores.append((path,sample_score))
        if save_extremes:
            if (sample_score['dice_avg'] < 0.8)|(sample_score['dice_avg'] >= 0.94):
                outputs = torch.nn.functional.sigmoid(outputs)
                outputs = torch.where(outputs > 0.5,1,0)
                mask = torch.squeeze(mask).cpu().detach().numpy()
                outputs = torch.squeeze(outputs).cpu().detach().numpy()
                sample_to_save = (mask, outputs)
                with open('test_sample_extreme/'+path.split('.')[0]+'.pkl','wb') as f:
                    pickle.dump(sample_to_save,f)
    scores = {}
    for key in sample_scores[0][1]:
        scores[key] = 0
        for sample in sample_scores:
            scores[key] += sample[1][key]
        scores[key] /= len(sample_scores)
    if return_sample_scores:
        return scores, sample_scores
    return scores

def tta_predict(model, image, do_flips):
    """
    Make an averaged prediction using test-time augmentation
    Parameters:
    -----------
    model : torch.nn.Module
        the model to use for prediction
    image : torch.tensor
        the batched and augmented image tensor
        shape (B=1, A, C, H, W, D)
    do_flips : list(list(bool))
        list of flip instructions for each image augmentation
        do_flips[i][j] is whether axis j on augmentation i should be flipped

    Returns:
    --------
    outputs : torch.tensor
        The average of model predictions, shape (B=1,C',H,W,D)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    image = torch.transpose(image, 0,1)
    num_augments = image.shape[0]
    
    # These will have shape (B,C,H,W,D) for outputs
    # and (B,C,H,W,D) for mask
    in_shape = list(image[0].shape)
    
    outputs = torch.zeros(in_shape).to(device)
    for aug_idx in range(num_augments):
        # image_aug has shape (B, C,H,W,D) and mask_aug
        # has shape (B,C,H,W,D) or (B,H,W,D)
        image_aug = image[aug_idx]
        outputs_aug = model(image_aug)
        for axis in range(len(image_aug.shape)-2):
            outputs_aug = torch.flip(outputs_aug,dims = (axis+2,)) if do_flips[aug_idx][axis] else outputs_aug
        outputs += outputs_aug
    outputs /= num_augments
    return outputs

def score(outputs,mask,metrics,metric_names, report_scores = False):
    """
    Compute and report metrics
    Parameters:
    -----------
    outputs : torch.tensor
        The predicted segmentation mask, shape (B=1, C', W, H, D)
    mask : torch.tensor
        The ground truth segmentation mask
        shape (B=1,C',H,W,D) or (B=1,H,W,D) depending on encoding
    metrics : list of torchmetrics metrics
    metric_names : list of names of metrics as strings

    Returns:
    --------
    results : dict
        keys are score types and value are corresponding scores
    """
    results = {}
    for metric_idx,metric in enumerate(metrics):
        metric_scores = metric(outputs,mask)
        for class_idx in range(3):
            if torch.sum(mask[:,class_idx,:,:,:]) == 0:
                output_preds = torch.nn.functional.sigmoid(outputs)
                output_preds = torch.where(output_preds > 0.5,1,0)
                if torch.sum(output_preds[:,class_idx,:,:,:]) == 0:
                    metric_scores[class_idx] = 1
                else:
                    metric_scores[class_idx] = 0
        for class_idx,suffix in enumerate(['et','tc','wt']):
            results[metric_names[metric_idx]+'_'+suffix] = metric_scores[class_idx]
            if report_scores:
                print(f'{metric_names[metric_idx]} score for class {suffix}: ... {metric_scores[class_idx]:.4f}')
        
        avg_score = torch.mean(metric_scores)
        results[metric_names[metric_idx]+'_avg'] = avg_score
        if report_scores:
            print(f'Average {metric_names[metric_idx]} score across classes ... {avg_score:.4f}')
    return results
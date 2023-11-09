import torch
from tqdm.auto import tqdm
import pickle
from lib.image_utils import unchunk_image
import pdb

def evaluate_loop(model, dl, metrics, metric_names, return_sample_scores = False, save_extremes=False, chunking = True, flipping = True):
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
    for image,mask,extra_data,path in tqdm(dl,leave=False):
        path = path[0]
        if chunking:
            outputs = chunk_predict(model, image, extra_data, flipping = flipping)
        else: outputs = tta_predict(model, image, extra_data)
        sample_score = score(outputs,mask,metrics,metric_names)
        sample_scores.append((path,sample_score))
        if save_extremes:
            if (sample_score['dice_avg'] < 0.8)|(sample_score['dice_avg'] >= 0.94):
                outputs = torch.nn.functional.sigmoid(outputs)
                outputs = torch.where(outputs > 0.5,1,0)
                mask = torch.squeeze(mask).cpu().detach().numpy()
                outputs = torch.squeeze(outputs).cpu().detach().numpy()
                sample_to_save = (mask, outputs)
                prefix = 'test_sample_extreme'
                if chunking: prefix+='_chunk'
                with open(prefix+'/'+path.split('.')[0]+'.pkl','wb') as f:
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

def chunk_predict(model, image, extra_data, flipping = False):
    if flipping: do_flips, overlaps = extra_data
    else: overlaps = extra_data
    image = image[0]
    outputs = model(image)
    if flipping:
        for idx in range(outputs.shape[0]):
            for axis in range(len(do_flips[idx])):
                unflip = torch.flip(outputs[idx],dims = (axis+1,)) if do_flips[idx][axis] else outputs[idx]
                outputs[idx] = unflip
    outputs = unchunk_image(outputs,overlaps)
    return outputs

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
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
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

def eval_model(prefix, valid_dupe_factor = 4, valid_batch_size = 1, chunking = True, flipping = True, save_results = False, print_status=False, filters = 16):
    """
    Compute evaluation scores for a trained model on both the validation and test sets
    Parameters:
    -----------
    prefix : str
        The model filename prefix
    valid_dupe_factor : int
        The number of times to duplicate the validation samples. Ignored if chunking = True
    valid_batch_size : int
        The batch size to use for validation samples.  Ignored if chunking = True
    chunking : bool
        Whether to use the chunking validation method
    flipping : bool
        Whether to flip axes randomly on chunks when using the chunking validation method.  Ignored if chunking = False
    save_results : bool
        Whether to save the score results as a file
    print_status : bool
        Whether to print the status
    filters : int
        Number of filters to use at the first stage of the UNet (should match what was used when training)
    Returns:
    --------
    results : dict
        Dictionary of scoring results
    """
    with open(prefix+'_split_info.pkl','rb') as f:
        train_paths, valid_paths, test_paths = pickle.load(f)
        
    valid_dl = get_dl(
        valid_paths, training = False,
        batch_size = valid_batch_size, dupe_factor = valid_dupe_factor,
        flipping = flipping, chunking = chunking,
    )
    
    test_dl = get_dl(
        test_paths, training = False,
        batch_size = valid_batch_size, dupe_factor = valid_dupe_factor,
        flipping = flipping, chunking = chunking,
    )
        
    model = UNet(dim=3, out_channels = 3, init_features=filters, num_stages=4)
    model.load_state_dict(torch.load(prefix+'_weights.pth'))
    
    dice = MultilabelF1Score(num_labels=3,average='none')
    accelerator = Accelerator(mixed_precision = 'fp16')
    model, valid_dl,test_dl, dice = accelerator.prepare(
    model, valid_dl,test_dl, dice
    )
    set_seed()
    results = {}
    results['model'] = prefix
    results['flipping'] = flipping
    
    model.eval()
    with torch.no_grad():
        if print_status: accelerator.print('Evaluating on test set...')
        test_metrics, sample_test_metrics = evaluate_loop(
            model, test_dl, [dice], ['dice'], return_sample_scores=True,save_extremes = True,
            chunking = chunking, flipping = flipping
        )
        
        if print_status: accelerator.print('Evaluating on validation set...')
        valid_metrics, sample_valid_metrics = evaluate_loop(
            model, valid_dl, [dice], ['dice'], return_sample_scores=True,save_extremes = True,
            chunking = chunking, flipping = flipping,
        )
        for result in [sample_test_metrics, sample_valid_metrics]:
            for sample in result:
                for key in sample[1]:
                    sample[1][key] = [float(sample[1][key].cpu().detach())]
        for result in [test_metrics, valid_metrics]:
            for key in result:
                result[key] = [float(result[key].cpu().detach())]
        results['validation'] = {
            'set':valid_metrics,
            'sample':sample_valid_metrics,
        }
        results['test'] = {
            'set':test_metrics,
            'sample':sample_test_metrics,
        }
        if print_status: accelerator.print('Done!')
    if save_results:
        filename = prefix+'_eval-results_flip.pkl' if flipping else prefix+'_eval-results.pkl'
        with open(filename,'wb') as f:
            pickle.dump(results,f) 
    return results

def sample_stats(results, dataset = 'validation', print_samples = False, print_stats = True):
    """
    Build dataframes of dataset summary statistics results and sample score results
    Parameters:
    -----------
    results : dict
        The dictionary of results output by eval_model
    dataset : str
        Which dataset to use.  Must be validation or test
    print_samples : bool
        Whether to print the sample score df
    print_stats : bool
        Whether the print the summary statistics df
    """
    sample_metrics = results[dataset]['sample']
    sample_score_df = pd.concat([pd.DataFrame(sample[1],index=[sample[0]])\
           for sample in sample_metrics]
).sort_values(by='dice_avg',ascending=False)
    stats_df = pd.DataFrame(data = {
        'mean':sample_score_df.mean(),
        'std dev':sample_score_df.std(),
        '25th perc':sample_score_df.quantile(q=0.25),
        '75th perc':sample_score_df.quantile(q=0.75),
    })
    if print_samples:
        print(f'{dataset} sample scores for {results["model"]}')
        print(f'flipping is {results["flipping"]}')
        print(sample_score_df)
    if print_stats:
        print(f'{dataset} score stats for {results["model"]}')
        print(f'flipping is {results["flipping"]}')
        print(stats_df)
    return sample_score_df, stats_df
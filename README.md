# Segmentation of brain tumor regions

**[View the project notebook as a HTML file](https://e-tweedy.github.io/BraTS2020.html)**

In this project, we implement a 3-dimensional UNet image segmentation model in order to predict brain tumor regions from MRI scan data.

We use the Training dataset from the 2020 BraTS (Brain Tumor Segmentation) Challenge, which ran in conjunction with the 23rd annual International Conference on Medical Image Computing & Computer Assisted Intervention (MICCAI).

Each sample in the dataset consists of 3-dimensional MRI images for native (T1), post-contrast T1-weighted (T1-CE), T2-weighted (T2), T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes as well as a ground truth segmentation mask indicating a class label for each voxel.  For our study, we omit T1 and only use T1-CE, T2, and FLAIR channels.

For more information about the challenge and the dataset, please visit the [BraTS 2020 website.](https://www.med.upenn.edu/cbica/brats2020/)

In order to experiment with the notebook, you need to first acquire the BraTS 2020 dataset, e.g. as a .zip file from [this Kaggle dataset item](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

## Model, training, and evaluation summary

* The model has a 3-dimensional UNet structure with four encoder levels and four decoder levels, withe group normalization after each convolution layer.
* Training samples are augmented using random Gaussian noise, random cropping, and random axis flipping in order to discourage overfitting and improve generalization.
* Segmentation masks are encoded from their original class labels to the more semantically meaningful classes of Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT).  Note that voxels can have several of these classes, so this is a **multilabel classification problem.**
* The model is trained for 30 epochs using the [Ranger21 optimizer](https://github.com/lessw2020/Ranger21) and a base learning rate of ``1e-3``.
* The loss function used during training is [GeneralizedDiceFocalLoss](https://docs.monai.io/en/stable/losses.html#generalizeddicefocalloss) implemented in [monai.losses](https://docs.monai.io/en/stable/losses.html), which is an unweighted sum of the Focal loss and Dice loss.
* Test-time augmentation is used when making predictions on the validation or testing set: each sample produced four augmented images, and predictions on these augmentations are averaged to produce a final prediction.
* The model predictions are evaluated using two metrics as implemented in [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/):
    * Intersection over Union (IoU) score, implemented as [MultilabelJaccardIndex](https://torchmetrics.readthedocs.io/en/stable/classification/jaccard_index.html)
    * Dice score, implemented as [MultilabelF1Score](https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html)
* The model attained the following average metric scores across the validation and testing sets:

<img src="results.png" alt="Evaluation results" width="800"/>

## Repository contents:

* ``BraTS2020.ipynb`` a Jupyter notebook in which training and evaluation is performed
* directory ``lib`` containing several modules used in the above notebook:
    * ``data.py`` which contains the custom torch.utils.data.Dataset class
    * ``image_utils.py`` which contains tools for manipulating and augmenting image samples
    * ``infer.py`` which contains inference and scoring functions
    * ``plot.py`` which contains functions for plotting samples
    * ``unet'py`` which contains our UNet model class
* empty directories ``images``, ``masks``, ``test_sample_extreme``, ``trained_model`` for housing saved data as needed
* ``requirements.txt`` file
* ``BraTS2020.html`` an exported HTML version of the Jupyter notebook

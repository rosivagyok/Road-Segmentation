# Road-Segmentation

Semantic segmentation of roads/lanes with Fully Convolutional Network based on Udacity project proposal.

Implements the model and method proposed by Long, Shelhamer & Darrell in 'Fully Convolutional Networks for Semantic Segmentation' 
on a road segmentation task/dataset.
(https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

# Arguments

--b_size (int)                Define batch size.

--b_per_ep (int)              Define batches per training iterations.

--ep (int)                    Define amount of training iterations.

--learning_rate (float)       Define learning rate.

--augmentation (bool)         Perform preprocessing in training.

--gpu (int)                   Which GPU to use.

run $ python main.py

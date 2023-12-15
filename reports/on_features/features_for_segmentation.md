
# Features for Image Segmentation

Our exp of features overfitted.

There are quite a few works that obtain per-pixel features to later do image segmentation.
Our motivation is overcoming the limitations of the methods we have proposed. On the one hand we have pure DINO which has nice semantic features but it doesn't really know about objects and the features are coarse. Therefore, even when doing upsampling, the ability to correctly segment objects misses the notion of objectness. On the other hand, SAM is all about objectness but it's promptable. To use it in a non-promptable way is impossible, and the closest to fully automated behavior  is synthesizing positive clicks on a grid. The result is a fixed list of masks that  can't be further modified. The makss can be classified but if they aren't correct in the first place whatever method built upon them is going to perform badly. 

Therefore we have, on the one hand, nice semantic features that don't know about objects nor segmentation, and on the other hand a promptable network that knows about objects but has no semantics nor flexibility. To merge both we need a network that knows about objects but that is more flexible. A feature space where objects are separated seems like a good way to achieve this.

The feature space where boject are separated can be multidimensional and it can be related to position or not. It seems important that, fr the use of convolutional networks and also to leverage translation invariance, the positions should be considered / added at the output. My original way of doing this was through scaling and concatenation, but this is not as elegant, interpretable and learnable as the paproaches that were proposed in a few papers. 

Notably, the original paper Semantic Instance Segmentation with a Discriminative Loss Function puts features inside balls. 
But there are other papers:
- Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images
    - learn embeddings such that the distance on the embedding space is equal to the distance on the image (differemt objects should average to 0 while same object features should represent location)
- END-TO-END PANOPTIC SEGMENTATION WITH PIXEL-LEVEL NON-OVERLAPPING EMBEDDING
    - a bad copy of the original paper
- Instance Segmentation of Biological Images Using Harmonic Embeddings
    - they predict a multidimensional "color" that is in fact the center position represented as frequencies
- Semantic Instance Segmentation via Deep Metric Learning
    - they predict vectors per pixel such that a simple classifier over their difference tells if they belong or not to the same object
- SGN: Sequential Grouping Networks for Instance Segmentation
    - a different approach, trying to detect breaks on horizontal and vertical directions and then doing something with them
- Deep Watershed Transform for Instance Segmentation (2017)
    - older but interesting, they predict directions inside objects first and rely on semantics
- Recurrent Pixel Embedding for Instance Grouping (2017)
    - They map pixels to a hypersphere of dim 64 and after apply mean-shift clustering in a differentiable way and train end-to-end
- Semi-convolutional Operators for Instance Segmentation
    - fancy math to say that cnns are translation invariant and as such you need some relative position instead of global (what I got from the figures)
- Pixel-level Encoding and Depth Layering for Instance-level Semantic Labeling
    - predict semantic, depth and direction to the center
- Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth
    - the new version of the original paper. They realize that the clustering width was important and that relative positions are also important (in fact I don't know how they managed before). The classifier is a gaussian and they optimize for IoU directly 
- The Best of Both Modes: Separately Leveraging RGB and Depth for Unseen Object Instance Segmentation
    - synthetic to real predicting depth and masks from depth first and then masks from the result and rgb
- Learning Category- and Instance-Aware Pixel Embedding for Fast Panoptic Segmentation
    - they map pixels to an embedding space with balls but also the embedding space is divided in classes therefore they do class and instance at the same time
- CLUSTSEG: Clustering for Universal Segmentation
    - they do convert the image to features but then they use seeds and an em like approach to solve the segmentation task
- OVeNet: Offset Vector Network for Semantic Segmentation
    - They refine a standard segmentaiton output with the offset vectors (positions) and some uncertainty map



# Experiments

Created simulated data containing random rectangles that many times overlap and that sometimes have the same color. The background is balck and all colors have the same mean. If rectangles touch then they don't have the same color. 
The network is a very small one with efficient vit blocks and pixelshuffle as decoder. 


logs:
- I trained a network with a basic loss, unbalanced, on 90 train features and 10 val features. The loss converged beautifully for both sets to a constant output. We see that the cosine annealing sometimes hurt, but not much.
    - training the same on more data shouldn't make much sense
- To avoid convergence to a constant I increased x100 the weight of the push loss. It also converged to the mean.
- The simplest hinge started to give something. Although with 90 training samples it's still not overfitting... I wonder if the capacity of the network is bad. I whould monitor all losses and the variance of the output at the same time. 
- In fact 90 training examples are less than that because I drop the last batch.


to-do:
[x] increase network capacity <- now using a small vitreg 
- monitor all losses and the variance of the output at the same time (in validation)
- log the images in color too
- code the other losses:
    - offset to learnable center
    - last paper of van gool (variable ball)
    - similarity-based classification
    - frequency centers
- test that all work ok and then launch one training per loss










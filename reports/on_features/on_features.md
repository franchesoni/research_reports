# On visual features

## Introduction
DINOv2 is a nice feature extractor but it was trained with some invariance to position, and what it encodes is mostly the position of the patches _relative_ to the objects in the image. DINOv2 feature space is very powerful, and we can think of it as if it were a more sophisticated color space.

On the other hand, SAM features aren't very discriminative but they can be queried by prompt vectors to extract masks. To some extent, the masks are present in these feature space and are just selected from it (excatly how is unknown to me). SAM was trained with masks and unlike DINOv2 it does rely on absolute positions.

There are three issues with these features:
1. SAM misses semantic information and DINOv2 misses the concept of mask
2. They are coarse (downsampling higher than 8x)
3. The features extracted from SAM aren't flexible: the mask generation is far from perfect and SAM doesn't work on the high click regime. Moreover, SAM isn't roobust to arbitrary clicks.

That's why we need to improve it and get a mask-based feature space. A feature space that knows the notion of object, part and subpart.

## The implications
Imagine we have a feature space that knows the concepts of object, part and subpart and can be used to generate masks, basically by clustering. This is, more or less clustered feature spaces would yield more or less dense image partitions.
This is the first application: fast unsupervised segmentation.

On top, interactive image segmentation can be achieved via graphcut or graph coloring. This is the second application. Note that here we solve important issues of prevailing IIS methods, namely robustness and consistency.

Now, equipped with these sophisticated position features, we can tackle the HILL problem. A basic methon would be to presegment the images and then refine the segmentation if needed. But in fact given predicted scores per patch we can compute a graph cut where two terms are considered: 1. the scores' clustering (similar scores in each group) and 2. the valueof the cut, where the distances are given by the positions.

## The difficulties

### Generalization
How do we learn such features? All IIS methods do it from segmentation masks. Where to find them? If we train over SA-1B, will we generalize beyond SAM?. If we don't overfit (which is unlikely), the answer is likely yes. The main contribution is not that much the semgentation performance but the way in which the segmentation is computed.

### Fitting
Will we be able to learn anything at all? How do we handle the ambiguity of the task? How exactly do we formulate our loss? It makes sense to ask, for each ground truth mask, to have the features inside be similar among them and different from the rest. Euclidean distance seems appropriate. On top, we can put a tanh in the end to restrict features to be in $[-1,1]$.

The minimum viable product is trying to fit only brightness with a sigmoid in the end to a handful of images only. We do need to treat each connected component as different. We want local superpixels, not semantically similar parts (that's for DINOv2).

### Computation and dimensionality
To make things efficient we can use EfficientViT. A small version could suffice, although the trend is to train big models and then distill to smaller ones. 

The main issue is how many features to predict. More are better, but they increase the memory. I was thinking about multi-resolution feature maps. These can be built in powers of 2. Giving x2 more importance to one map x2 less resolved implies having x4 more features. This means that the number of features at each resolution is the same. That should work.


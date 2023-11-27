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

### Related work
Enzo Ferrante suggested me to look at _"Semantic Instance Segmentation with a Discriminative Loss Function"_, a paper from CVPR2017 which does something very similar to what I wanted to do. There are a few key points:
- their loss is not only constrative but also regularizes the output and adds hinge loss, i.e. it only acts on a ball. The hinge part is a great idea.
- the features are full resolution and 8 in number
- they do instance segmentation for which they previously rely on semantic segmentation to handle multiclasses
- they suppossedly can do (but not try) classification or semantic segmentation via doing the contrastive loss on a mini-batch in a class-consistent way

Takeaways: 1. triplet hinge loss, 2. train on a dataset of instances


## Minimum viable product (MVP)

We'll use EfficientViT. There are other efficient transformer backbones around but this is especially good in segmentation and was trained using segment anything as target. On top it's supposed to handle high resolution images (because of the downsampling to 512x512). 

1. clone the efficientvit repo (around 50MB)
`git clone https://github.com/mit-han-lab/efficientvit.git`
2. download weights for smaller SAM-like model and move it to (`assets/checkpoints/sam`)
`gdown --id "1AiaX67kT-TX5yr0wOZn51jICj-k5aBmx"`
     82 Expects a numpy array with shape HxWxC in uint8 format.

## The plan (22 Nov)

Here's the plan:
- **dataset:** use SA-1B, a big dataset with okish masks and natural images. The masks corresponds to objects, parts and subparts, but they don't cover the whole image. This dataset should serve as baseline and finetuning could be done later with ood datasets or some that present finer masks.
- **architecture:** use an efficientvit (which is the most modern and performing architecture) followed by some simple mbconv decode stages. For each decode stage we pixelshuffle to map to a specific resolution. We don't use all the channels because "Vision Transformers need Registers", but only multiples of 3 (3 for full res, 12 for x2 ds res, and so on...). Each resolved feature map is followed by a sigmoid, making features be colors in [0,1]
- **loss:** as in the instance seg paper, the loss will be a pull loss for mask features too far away from the avg mask feature, and a push loss for mask centers (or features corresponding to different masks) too close to each other. There should be one loss for each resolution with appropriate weighting (less resolved counts more)
- **engineering:** train on a 8-gpu node.
- **inference:** clustering can be done at different resolutions. Each resolution gives us a different distance mesh that could be combined. This way we can do clustering or other distance-based stuff


## Minimum viable product (MVP) 2: gray-masks

Say we have a network, starting or not from sam and trained on rectangles of different sizes and colors. What we would like to do is to create a grayscale image where the rectangles have different gray levels. The solution should be a simple operation mapping colors to grayscale. The main thing is that we don't really care how close the rectangles are in color space, but we always map them to be separate in luminance space. If we e consider 10 gray

## Experimental results and reflections

1. We tried training a network on a piece of SA-1B dataset, generating samples comprising images and the 10 biggest masks for that image. The network is an efficientvit with a decoder head that are some simple blocks and pixel shuffling. The loss has a pull-if-far component between features of the same masks, push-if-close component between features of different masks, and push-if-close component between averages of features over masks. This was trained on weird over one gpu with 12GB of memory and full utilization. Loss was vectorized. Batch size is 8. Output dimension is 4 (more than enough). 
2. The results showed some segmentation was being done. My worry about the ordering of the coloring doesn't really seem important. Two of the four channels are useless. Proposed segments are relatively big. There is some evidence that the network could handle ambiguity correctly. However, the objectness doesn't seem to extend beyond SAM. It doesn't really work on satellite images and it proposed relatively big masks. Of course, training on more diverse datasets and with multiple object sizes makes sense. The main issue now is that the network is not really building a hierarchy of objects, which could be desirable. Should we do it synthetically?

I'm proposing a mask proposal generator. For that I need ground truth masks. It's hard to imagine the proposal generator generalizing to masks different than those seen as ground truth. It seems unavoidable to do some sort of self-supervision, in this case generating some target masks automatically. What was the hope? That because the target masks were unknown the method would account for similarity of masks that overlap automatically. This seems to happen, but the method doesn't really capture a hierarchy of masks like a fractal as I would have liked. 

There are two potential ways that come to mind to do this. The first one is to create different radius for balls, so that we know more or less that shared surfaces between masks should be at some specific distance. The second one is to look at different dimensions and ask, for instance, for pixels of two masks to be similar on one dimension while far away in another. None of these is satisfactory at first sight. How do I enforce a fractal structure in the loss? Maybe using different radius for different dimensions. The larger the radius the coarser the segmentation? Probably a loss like SAM's, which only takes the loss on the minimum per prediction (in this case per channel), would be best. The question here is how do we handle different resolutions and so on. Maybe training on a large scale dataset would be enough.






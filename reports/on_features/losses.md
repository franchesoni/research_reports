# Losses
If we want to solve image segmentation with a bottom-up approach we need to map every pixel to a new vector such that similarity-based methods (grouping, classifying) work well. 

Image segmentation poses serious challenges:
- multiscale / fractal decomposition of the image
- ambiguity of the segmentation (traditionally an image segmentation is seen as a tree, but this doesn't account for transparencies or reflections or matting due to blur)

We don't aim to solve these, but simply to provide a feature space where segmentation can be more easily achieved and users can build upon. Superpixel algorithms such as SLIC (which is basically k-means with a reduced spatial context window) are ok and greatly flexible, but not very good at knowing that objects might sometimes have different colors. We plan to overcome this by learning.

## Notation
- $I$ the input image of shape $(H, W, 3)$. As a function, $I: \Omega \to [0,1]^3$ and $\Omega$ is the spatial dimension $[1, \dots, H] \times [1, \dots, W]$.
- $y_M$ a set of $M$ binary masks each corresponding to an object on the image $I$ (usually $2 < M < 10$). Note that $m \in y_M$ is one such mask.
- $f$ is the network and $\hat{y}$ its output. The shape of $\hat y$ is $(H, W, F)$.
- $\hat{y}_c$ is the $c$-th channel of the output. $\hat{y}_c$ can be indexed with a mask $m$ to get the set of output pixels inside the mask $\hat{y}_c[m]$.
- Note that each pixel in the image $I$ can also be seen as an $[0,1]^5$ vector where the first three components are RGB and the last two are the position in terms of normalized rows and columns.
- $[\cdot]_+$ is the hinge function, i.e. $\max{(0,x)}$. 
- $\mu_m$ or $\mu_m[c]$ is the mean of pixel features in mask $m$ (considering all or one channel, respectively)
- $x$ is any position on the image indexing an individual pixel, i.e. $x\in \Omega$


## Losses
### Push-pull centroid loss
The simplest and kind of naive loss is to simply ask for similar features to be close together and different features to be far appart. This is

**Attraction**
$$L_A(\hat y, y_M) = \frac 1 M \sum_{m \in y_M} \left( \frac 1 {|m|} \sum_{z \in \hat{y}[m]} ||\mu_m - z||\right)$$

**Repulsion**
$$ L_R(\hat y, y_M) = - \frac 1 M \sum_{m \in y_M} \left( \frac 1 {|\Omega|-|m|}\sum_{z \notin \hat{y}[m]} ||\mu_m - z|| \right)$$
**Full loss**

$$L(\hat y, y_M) = \alpha L_A(\hat y, y_M) + L_R(\hat y, y_M) $$



**Parameters**
- $\alpha$: the importance of the attraction loss over the repulsion loss. If close to zero diverse features are encouraged.

**Comments:**
This loss has as issue the contradictory objective. Its behavior is controllable with a parameter $\alpha$. To make it give diverse solutions $\alpha$ should be very low and pixels should be discriminable from each other. 

### Push-pull balls loss 
This is an improvement on the previous loss by adding balls where the forces act via hinge losses. 

**Atraction**

$$L_A(\hat y, y_M) = \frac 1 M \sum_{m \in y_M} \left( \frac 1 {|m|} \sum_{z \in \hat{y}[m]} [||\mu_m - z|| - r_A]_+ \right)$$

**Repulsion**
$$ L_R(\hat y, y_M) = \frac 1 M \sum_{m \in y_M} \left( \frac 1 {|\Omega|-|m|}\sum_{z \in \hat{y}[m^c ]} [r_R - ||\mu_m - z||]_+ \right)$$

**Full loss**
$$L(\hat y, y_M) = \alpha L_A(\hat y, y_M) + L_R(\hat y, y_M) $$

**Parameters:**
- $\alpha$: the importance of the attraction loss over the repulsion loss. If close to zero diverse features are encouraged.
- $r_A$: radius of attraction ball. Being higher involves less number of segments in the image. Being smaller involves more constant features on some regions, even if they present hierarchy.
- $r_R$: radius of repulsion ball. Being higher involves less number of segments in the image. Being smaller implies that segments will not necessarily be easily distinguishable from each other, but it's relative to $r_A$. It's usually $r_R = C r_A$ with $C=2$ or $C=3$. 

### Push-pull balls loss (paper)
Semantic Instance Segmentation with a Discriminative Loss Function https://arxiv.org/pdf/1708.02551.pdf

**Attraction** is the same than above, but with a square

$$L_A(\hat y, y_M) = \frac 1 M \sum_{m \in y_M} \left( \frac 1 {|m|} \sum_{z \in \hat{y}[m]} [||\mu_m - z|| - r_A]_+^2 \right)$$

**Repulsion** is different as it is between masks' centroids:
$$ L_R(\hat y, y_M) = \frac 1 {M(M-1)} \sum_{m_1, m_2 \in y_M; m_1\neq m_2} [2r_R-||\mu_{m_1}-\mu_{m_2}||]_+^2$$

**Regularization** pulls everything to $0$
$$L_{reg}(\hat y, y_M)= \sum_m ||\mu_m||$$

**Full loss** is simply
$$L(\hat y, y_M) = \alpha L_A(\hat y, y_M) + \beta L_R(\hat y, y_M) + \gamma L_{reg}(\hat y, y_M) $$

**Parameters:**

$r_R=1.5, r_A=0.5, F=8, \alpha=1, \beta=1, \gamma=0.001$.
For most experiments $r_R = 3 r_A = 1.5$, and that the network doesn't use a sigmoid at the output. Also, they used output dimension $F=16$ (CVPPP i.e. plant leaves) and $F=8$ (Cityscapes). Also they use a pretrained ResNet for semantic segmentation. 

The network they train for instance separation is trained for all classes independently. The key differences between what we tried to do on the last loss and this paper are two: 1. we don't use any semantic labels nor semantic segmentation and 2. we involve all pixels on the repulsion loss, as we assume all must belong to separate segments.


### Logit difference classification loss (paper)
Semantic Instance Segmentation via Deep Metric Learning
https://arxiv.org/pdf/1703.10277.pdf

**Similarity function**
Given positions $x_1, x_2 \in \Omega$, the similarity function is:
$$\sigma(x_1, x_2) = \frac{2} {1 + \exp(||\hat y (x_1) - \hat y (x_2)||_2^2)}$$

**Final loss**
Sample a set of pixels $S$ comprised by randomly sampling $K$ pixels from each object mask. Then the loss is a weighted cross entropy:
$$L(\hat y, y_M) = - \frac 1 {|S|} \sum_{x_1, x_2 \in S} \sum_{m \in y_M} 1_{\{x_1 \in m\}} \frac 1 {|m|} \left[ 1_{\{ x_2 \in m \}}\log{(\sigma(x_1, x_2))} + 1_{\{ x_2 \notin m \}}\log{(1-\sigma(x_1, x_2))}
\right] $$

**Comments**
They use another head that produces a seed heatmap for different sizes, handling multiscale in this way. However the method to obtain seeds is a little too long to explain here, but could be done if needed.

**Parameters:**
$F=64$
$K=10$


### Offset loss (paper)

**Components**
This loss is simple. There are three predicted segmentations, $\hat{y}_{s}, \hat{y}_f, \hat{y}_i$, based on three predicted quantities, namely the offset map, the uncertainty map and the class scores, respectively. The offset map $o:\Omega \to [0,1]^2$ helps generate a prediction such that $\hat{y}_s(x) = \hat{y}_i(x+o(x))$. The prediction associated to the uncertainty map $U$ is the weighted average $\hat{y}_f = (1-U) \hat{y}_i + U \hat{y}_s$. The loss is the sum of the losses achieved by each of the three segmentations.
If the ground truth segmentation is $H$ then the loss is 
$$L_{\text{semantic}} = L(\hat{y}_f, H) + \kappa L(\hat{y}_s, H) +  \lambda L(\hat{y}_i, H)$$
where $L$ is "Ohem Cross-Entropy Loss".

**Uncertainty loss**

The uncertainty map involves a second loss component which is 
$$L_f(x) = - \left[1_{H(x)= H(x+o(x))} \log(U(x)) +  1_{H(x)\neq H(x+o(x))} (1-\log(U(x)))\right].$$

**Final loss**
$$L_{\text{final}} = L_{\text{semantic}} + L_f$$

**Parameters**
$\tau= \lambda=  0.5$
where $\tau$ is the maximum offset vector length as restricted by a tanh layer. (probably $\kappa=0.5$ too but they report $\mu$ instead, probably a typo).

### Offset + sigma (paper)
Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth
https://openaccess.thecvf.com/content_CVPR_2019/papers/Neven_Instance_Segmentation_by_Jointly_Optimizing_Spatial_Embeddings_and_Clustering_Bandwidth_CVPR_2019_paper.pdf

A mix of using balls and offset vectors. The offset deals with the translation equivariance problem. The balls are of adaptive size to handle objects of different sizes.

**Network output**
The network $f$ processes all the image $I$ at once but provides the same resolution at the output. In our case an output pixel at position $x$ is $f(I)(x) = (o, \sigma)$, from where we could also write that the offset for $x$ is $o(x)$ and its sigma $\sigma(x)$. The final prediction $z$ is considered to be augmented with the position, this is $\hat{y}(x) = (x+o(x),\sigma(x))$. We also consider $\mu_m = \frac 1 {|m|} \sum_{x\in \hat{y}[m]} x + o(x)$ and $\sigma_m = \frac 1 {|m|} \sum_{x\in \hat{y}[m]} \sigma(x)$

**Probability of belonging to an instance**

Given a centroid $\mu_m$ associated to mask $m$ we have that the probability of pixel $x$ to belong to the same class as the centroid is 

$$\phi_m(x) = \exp{\left(- \frac{||x-\mu_m||^2 }{2\sigma_m^2} \right)}.$$

**Segmentation loss**
The loss between the "prediction" (to be thresholded at 0.5) and the ground truth mask is the Lovasz-hinge loss, call it $L$. They say this $L$ is a convex surrogate of the IoU. The implementation is given in 
https://github.com/bermanmaxim/LovaszSoftmax .

**Seed map**
To do further clustering a seed map $s$ is created. This seed map is trained by minimizing the prediction error against the output of the gaussian. Therefore 
$$L_{\text{seed}} = \sum_{m\in y_M} \sum_x ||s(x)-\phi_m(x)||^2$$
but we don't backpropagate through $\phi$.

**Smoothing $\sigma$**

We would like to have one constant $\sigma$ per region, therefore
$$L_{\text{smooth}} = \frac 1 {|m|} \sum_{x\in m} ||\sigma(x) - \sigma_m||^2$$

**Final loss**
$$L(\hat{y}, y_M) = L(\hat{y}, y_M) + L_{\text{smooth}}  + L_{\text{seed}}$$

**Code**
More details can be found in the code https://github.com/davyneven/SpatialEmbeddings .

## Other papers
**Learning Category- and Instance-Aware Pixel Embedding for Fast Panoptic Segmentation**: some sort of triplet loss like before but also with semantic classes. Cosine similarity. 

**Pixel-level Encoding and Depth Layering for  Instance-level Semantic Labeling**: they predict the center directions as an auxiliary objective

**The Best of Both Modes: Separately Leveraging RGB and Depth for Unseen Object Instance Segmentation**: they predict the center directions as an auxiliary objective

**Recurrent Pixel Embedding for Instance Grouping**: they use cosine similarity and a hinge only on the repulsion part. 

**Semi-convolutional Operators for Instance Segmentation**: haven't read

**Deep Watershed Transform for Instance Segmentation**: seems too far away and too complicated

**Instance Segmentation of Biological Images Using Harmonic Embeddings**: they regress the center but the center is expressed as frequency features.



**CLUSTSEG: Clustering for Universal Segmentation**: they need queries, superpixels are worse than SLIC, what?

**Segment Anything**: a promptable segmentation model. You put a positive click, you get three masks. If you put clicks everywhere and postprocess the masks you can get a segmentation of the image.


## High level overview

Of all the losses, the offset loss isn't appropriate because it's for segmentation with full supervision.

All the rest could be formulated with only one mask per image. In fact we should check that we overfit and then start adding more data until the network is forced to generalize.

Some of them require an activation function (e.g. sigmoid) to control the output while others don't. Because of this variation we'll put the activation function into the loss function. 

Some of them involve some random sampling when others don't. A formulation I find general is to predict the residual for position and for color and maybe a couple of other vectors too.

Therefore the losses involve images and masks and we should overfit one mask per image (or more) before increasing the number of examples. 


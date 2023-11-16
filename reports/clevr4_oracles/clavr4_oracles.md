# On binary classification oracles over CLVR4
## Teaser

Consider the following two dimensional samples with the respective labels (positive O, unlabeled ?, negative X).
```
O---?
|   |
O---X
```

How would you classify the last label? Similarity-based methods would give a different answer to a linear classifier. What is correct? Of the possible ways of labeling `?`, which is the one that yields the least complex labeling? 


## Dataset
CLVR4 is a dataset comprising synthetic images of objects generated based on different choices of color, shape, texture and count.
For a single image, there is only one color, shape, texture and object count.
The options available for each of the 4 dimensions are 10.
The dataset was initially introduced to study Generalized Category Discovery (GCD) but in fact is useful to study classification over different semantic dimensions in general.
For our purposes, all images $I\in D$ can be described as 4 one-hot codes of length 10 or equivalently, four indices $I = (x_1, x_2, x_3, x_4) \in [0, 9]^4$.
Because we'll assume that the options on each dimension are unordered (the only one that has a natural order is "count"), we can say that the distance $d(I_i, I_j)$ between two images $I_i$ and $I_j$ is the number of dimensions in which the indices differ, i.e. $d(I_i, I_j) = \sum_{1\leq k \leq 4} 1_{{I_i}_k \neq {I_j}_k}$. Given this distance we can build a graph, and we lose no information if we only consider the edges between nodes that are $d=1$ appart. Note that this distance is a metric.

## Our problem
We want to find what's the best classication we can predict if we are given binary labels over some images. We will assume that the binary labels correspond to a target labeling. A target labeling would be basically a set of images of arbitrary cardinality. In other words, a target labeling $y$ assigns to each image $I_i \in D$ a ground truth label $y_i \in \{-1, 1\}$.
Note that there are simpler or more complex labelings. We'll build a notion of complexity later.
At first we will assume that the images are annotateed at random. Note that active learning could greatly speed up the obtention of a correct classifier.

notes: we could also assume that the number of categories in each dimension isn't known a priori. Not only that but we could assume that some subsets of categories have smaller distance, e.g. odd and even counts. These extensions might culminate with the learning of a similarity metric, i.e. the deformation of the initial feature space. We will stick to our simpler but defying case.

## An heuristic predictor
Consider a predictor that assigns to each image $I_i$ a label $\hat{y}_i \in \{-1, 0, 1\}$ where $0$ means "I don't/can't know".
We want to find the oracular predictor, e.g. the best possible. We're assuming that the predictor has access to the exact value of texture, color, shape and count for each image. This is a very big assumption and it isn't usually true that the predictor has access to the exact value on the relevant dimensions in which the data varies.
We will assume that the predictor (and the oracle) follow Occam's Razor. This is, that the least complex solution that fits the data is the correct one.

One desired trait of the predictor is to treat all annotated samples the same (because they're random and not actively chosen). This symmetry reduces the options.

### Similarity-based methods
We can now classify the rest of the images depending on the number of elements in common with the annotated image. Depending on the number of elements in common (the inverse of a distance) we should set a zone of "similar" (positive images), "gray" ("I don't know") and "distinct" (negative). The classification thresholds should be optimized in order to reduce the complexity of the prediction. Similiarity-based methods such as k-nn converge asymptotically to the optimal Bayes generalization error, but this doesn't mean that they're efficient in learning. In fact, their efficiency very much depends on the distance metric. Sometimes a simple linear projection is better. 

### Learning
The simplest parametric method is argueably the linear classifier. It distorts the distance function by giving different importances to different dimensions.

### Complexity of a labeling
A ground truth labeling and a prediction can be more or less complex. If seeing the data as a graph, the complexity of the labeling is the complexity of the coloring of a graph. There are many options that could be considered: 
    - Edge Cut: In graph theory, an edge cut is a set of edges that, if removed, would divide a graph into two disjoint subgraphs. In the context of our binary labeling, we could consider the complexity of a labeling as the number of "cross-label" edges - edges that connect nodes of different labels (positive/negative). A simpler labeling would have fewer such edges, implying that similar images (in terms of the attributes) are more likely to share the same label. The drawback is that unbalanced labelings are simpler under this complexity criterion.
    - Cluster Homogeneity: This metric would assess the homogeneity within clusters of similarly labeled images. A simple labeling would result in highly homogeneous clusters, where nodes within the same cluster (same label) share a high degree of similarity. This could be quantified, for example, by the intra-cluster edge density.
    - Graph Modularity: Modularity is a measure of the structure of networks or graphs. It measures the strength of division of a network into modules (or clusters). High modularity means that the network can be easily divided into such modules with dense connections internally and fewer connections between modules. In our case, a simpler labeling might correspond to a higher modularity where positive and negative labels form two distinct clusters.
    - Minimum Description Length (MDL): This is a more information-theoretic approach, where the complexity of a labeling is related to the length of the description (or encoding) of the graph coloring. A simpler labeling would require a shorter description. This involves considering both the number of different labels and the regularity or patterns in the labeling. However, it's hard to get such description.
    - Label Smoothness: This metric would look at the smoothness of the labeling across the graph. A simple labeling would have large regions of the same label with fewer changes or transitions between labels.
    - Entropy: In information theory, entropy is a measure of the unpredictability or randomness of information content. Lower entropy would indicate a more predictable (and hence simpler) labeling scheme. However, we lack a probability distribution.

# Conclusion

An optimal classifier isn't well defined in a deterministic framework where dimensions represent actual semantically different attributes. The CLEVR4 dataset illustrates this issue. 
We can represent the dataset as a graph where distances correspond to semantic differences. Given some labels, similarity-based methods and linear classifiers provide different answers. On top, there isn't a single definition of what's complexity. Given a definition of complexity, we could define the optimal classifier as the one predicting _least complex_ labeling that fits the data. When a sample can't be classified as positive or negative (because of symmetry) we should paint it as gray.



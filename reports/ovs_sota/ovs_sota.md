
# Open Vocabulary Segmentation (OVS) state-of-the-art

Here I quickly lay down what are the best methods for open-vocabulary segmentation.
The MESS benchmark [1] evaluates many methods on many datasets, and CAT-Seg [2] comes clearly on top.
As of Nov 22 2023, only 10 papers cite CAT-Seg. In most of them, CAT-Seg is still extremely competitive.
There are two notable works that build upon and improve CAT-Seg by improving the CLIP encoder via self-distillation.
These are:
|Paper|Has code| Is Best |
|---|---|---|
|CLIPSELF: VISION TRANSFORMER DISTILLS ITSELF FOR OPEN-VOCABULARY DENSE PREDICTION [3] | :white_check_mark: | :x: |
|SILC: Improving Vision Language Pretraining with Self-Distillation [4] |  :x: | :white_check_mark: |


## Conclusion
Therefore what I'd recommend is for you to use SILC if the code has been released, CLIPSELF if not, and if the last one is too hard just go for plain CAT-Seg.

## References
1. MESS – Multi-domain Evaluation of Semantic Segmentation
2. CAT-Seg : Cost Aggregation for Open-Vocabulary Semantic Segmentation
3. CLIPSELF: VISION TRANSFORMER DISTILLS ITSELF FOR OPEN-VOCABULARY DENSE PREDICTION 
4. SILC: Improving Vision Language Pretraining with Self-Distillation 


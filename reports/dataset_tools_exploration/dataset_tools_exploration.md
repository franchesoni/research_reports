
# Exploration of python package `dataset_tools` for image segmentation

Supervisedly is a company that shared a python package and a bunch of computer vision datasets. 
This makes it easier for researchers to configure and deal with the data.
In this report we'll try to use this tool and create a benchmark with many image segmentation datasets.

Our requirements are:
    - high quality masks (visual comparison)
        - can I see an annotation error on some random 10 images of the dataset? (with a seed for reproducibility) if not, it's high quality
    - manageable size (<5GB)
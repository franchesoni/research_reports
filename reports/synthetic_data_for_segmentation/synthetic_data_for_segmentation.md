
# Synthetic data for image segmentation

Synthetic datasets are nice because they help us better explore the limitations of the models.

Here we explore a list of datasets for image segmentation. The datasets are ok if 1. they can be downloaded automatically and 2. they have train / test splits. Our conclusion is that the python package `dataset_tools` of Supervisedly is the easiest option to handle datasets, some of which are synthetic.

## Datasets

### OK
- SYNTHIA dataset 
    - train split `http://synthia-dataset.net/download/1135/`
    - test split `http://synthia-dataset.net/download/1137/`

- Supervisely Synthetic Crack Segmentation Dataset
    - available through `dataset_tools`

- Synthetic and Empirical Capsicum Annuum Image Dataset
    - `https://figshare.com/ndownloader/articles/12706703/versions/1`
    - `https://figshare.com/ndownloader/files/24061061`

### Maybe

- Synthetic Arabidopsis Dataset
    - `AWS_ACCESS_KEY_ID=ANPWP08W6RS2FG6LRZQL AWS_SECRET_ACCESS_KEY=dYyM4+LIc8jIXG3hY0SxVaavJ4zwyZTpIZkliYc2 aws s3 cp --endpoint-url https://s3.data.csiro.au --recursive s3://dapprd/000034323v004/ .`
    reason:
    - don't know if it has splits

- replicAnt - Plum2023 - Segmentation Datasets and Trained Models
    - `https://zenodo.org/records/7849570/files/replicAnt_Plum2023_Semantic_Segmentation.zip?download=1`

- SynthAer - a synthetic dataset of semantically annotated aerial images
    - `https://figshare.com/ndownloader/files/13025453`

### discarded

- Image-Bot: Everyday/Industrial Objects
    - has direct download link
    reason:
    - it's not really synthetic
    - doesn't have train/test splits

- Glasses Segmentation Synthetic Dataset
    reason:
    - hosted in kaggle `https://www.kaggle.com/datasets/mantasu/glasses-segmentation-synthetic-dataset/`

- Synthetic Whole-Head MRI Brain Tumor Segmentation Dataset
    reason:
    - 3D segmentation

- Synthetic Images for the Semantic Segmentation of Robotic Instruments in a Head Phantom
    reason:
    - requires log in

- Synthetic Operating Room Table (SORT) Dataset
    reason:
    - too big

- Synscapes: A Photorealistic Synthetic Dataset for Street Scene Parsing
    reason:
    - too big

- PeopleSansPeople: A Synthetic Data Generator for Human-Centric Computer Vision
    reason:
    - too big

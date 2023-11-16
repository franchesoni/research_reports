
# Exploration of python package `dataset_tools` for image segmentation

Supervisedly is a company that shared a python package and a bunch of computer vision datasets. 
This makes it easier for researchers to configure and deal with the data.
In this report we'll try to use this tool and create a benchmark with many image segmentation datasets.

We find that in many of these datasets only one class is present. Furthermore, in some of them the masks are only rough.

## Requirements
- ok quality masks (visual comparison)
    - can I see an annotation error on some random 5 images of the dataset? (with a seed for reproducibility) if not, it's ok
- easy
    - I should be able to understand the concept and segment the images correctly in my mind
- manageable size (<=5GB)

## Dataset list
A list of all available datasets can be found by using a `breakpoint()` just before a `dataset_tools.download('dummy')` call. When inside `download` go ahead until the `data` dict is loaded and they `print(data.keys())`. These are the datasets. (see permalink `https://github.com/supervisely/dataset-tools/blob/2b536ce2df06f840a2c1b5ea898308e441fcb14e/dataset_tools/repo/download.py#L139C15-L139C15`)

However, here all datasets are present. We want only those related to segmentation. Their webpage has a nice way to explore them.
We filter by "image", "mask" and sort by "size". Now we start looking at each one of them and classifying them in accepted or discarded.

### Accepted
- Accurate Drone Shapes Dataset <- very small objects!
- Accurate Nevus Shapes
- AeroScapes Dataset
- Alfalfa Roots <- fine structures
- Carrot-Weed
- CHASE DB1
- Coffee Leaf Biotic Stress Dataset
- Concrete Crack Segmentation Dataset
- FPIC-Component Dataset <- has empty masks!
- GlaS@MICCAI'2015: Gland Segmentation
- KolektorSDD2 Dataset  <- has empty masks!
- MangoNet
- MSD: Mobile Phone Defect Segmentation Dataset
- Panoramic Dental X-rays Dataset <- spatially predictable 
- Skin Cancer (HAM10000)
- Supervisely HRDA Plants Demo Dataset <- used for semi-supervised learning, training set is only 62 images, and created by supervisedly
- Synthetic Crack Segmentation <- only unstyled split
- Synthetic Plants Dataset
- Water Meters Dataset 
- WGISD Dataset <- has empty masks!
- PolypGen
- SUIM Dataset
- CIHP
- Full Body TikTok Dancing
- EMPS: Electron Microscopy Particle Segmentation




### Discarded

#### so-so quality
- Makassar Road <- one discontinuous line wasn't divided
- EDD2020 Dataset (disgusting)
- PASCAL VOC 2012 Dataset 
- Chest Xray Masks and Labels Dataset
- BBBC041Seg
- WaterDataset
- CitySegmentation <- images are very big and masks aren't pixel perfect (they're polygons)

#### bad quality
- Intraretinal Cystoid Fluid <- different annotations for the same image
- Semantic Segmentation Satellite Imagery
- CCP: Clothing Co-Parsing
- COCO-Stuff 10k
- Severstal
- Fetal Head UltraSound <- segmentations are elipsis
- CaT Dataset <- rough masks / regions
- Breast Ultrasound Images Dataset
- MinneApple
- Industrial Optical Inspection Dataset <- weak supervision
- MVTec LOCO AD Dataset
- PRMI
- EWS Dataset
- MVTec AD Dataset: annotations don't correspond to objects
- Tree Binary Segmentation Dataset
- Teeth Segmentation on Dental X-ray Images Dataset
- Fluorescent Neuronal Cells Dataset
- PLD-UAV Dataset
- DeepGlobe 2018 Road Extraction
- CNAM-CD
- CCAgT
- Urban Street: Branch
- DeepGlobe Land Cover 2018
- CelebAMask-HQ

#### other
- Pascal Context <- failed download
- Photometric Stereo Leafs <- too repetitive
- Plasmodium Falciparum from Images of Giemsa for Malaria Detection <- not found
- Self Driving Cars <- download failed
- Cracks and Potholes in Road <- too fine (hard), some little mistakes, segmentations only in a subpart of the image (hard)

#### too hard
- Heat Sink Surface Defect Dataset  <- it might be bad quality or I just don't get the concept
- Danish Golf Courses Orthophotos <- segmentes are even hard for me to guess
- CosegPP Dataset (ds structure is not straighforward)
- cwfid
- BTAD Dataset (I don't understand)
- Annotated Quantitative Phase Microscopy Cell Dataset (I don't see)
- AFID <- defects in clothing? hard

#### too big (>5GB)
- Sugar Beets 2016 Dataset
- Apple MOTS Dataset
- PV Dataset
- LoveDA Dataset
- Urban Street: Tree Dataset
- Supervisely Persons Dataset
- UAVid Dataset
- HyperKvasir Images Dataset
- Maize Whole Plant Image Dataset

#### lack of automated download due to license
- Plant Growth Segmentation Dataset
- COVID-19 Dataset
- Massachusetts Buildings Dataset
- Hoofed Animals Dataset
- Hacking the Human Body 2022 Dataset
- SIIM-ACR Pneumothorax Segmentation 2019 Dataset
- StrawDI_Db1 Dataset
- Semantic Drone Dataset
- DUTS Dataset
- Multi-Class Face Segmentation
- Eyes Microcirculation Dataset
- PH2 Dataset
- CaSSeD: Real World Data Dataset
- AIRS Dataset
- Alabama Buildings Segmentation
- Urban Street: Trunk Dataset
- FloodNet 2021: Track 1 Dataset
- ISIC 2017: Part 1 - Lesion Segmentation Dataset
- 38-Cloud Dataset
- Corridor Floor Segmentation Dataset
- Carvana Image Masking 2017 Dataset
- Urban Street: Leaf Dataset
- CBIS-DDSM Dataset
- Lips Segmentation Dataset
- RuCode Hand Segmentation 2021 Dataset




























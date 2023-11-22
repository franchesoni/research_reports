from pathlib import Path
import tqdm
from pprint import pprint
import shutil
import random
import os
import zlib
import base64
import json

from PIL import Image
import dataset_tools as dstools
import numpy as np
import cv2
import skimage


available_datasets = ['PASCAL VOC 2012', 'MVTEC D2S', 'BSData', 'CHASE DB1', 'Coffee Leaf Biotic Stress', 'MinneApple', 'PASCAL Context', 'LaboroTomato', 'tomatOD', 'Sweet Pepper', 'RoCoLe', 'LoveDA', 'Defects in Power Distribution Components', 'GC10-DET', 'Road Vehicle', 'PCB Component Detection', 'Full Body TikTok Dancing', 'Disease Detection in Fruit Images', 'HyperKvasir Segmentation', 'Road Damage', 'Water Meters', 'Synthetic Plants', 'Chest Xray Masks and Labels', 'CWFID', 'Weed Detection', 'Apple MOTS', 'Car License Plate', 'Tomato Detection', 'DeepBacs E. Coli', 'KolektorSDD2', 'PlantDoc', 'deepNIR Fruit Detection', 'Damage Detection of Power Plants', 'Cocoa Diseases', 'Rice Disease', 'WGISD', 'Wind Turbine Detection (by Luke Borkowski)', 'Cityscapes', 'Maize Cobs', 'Pylon Components', 'STN PLAD', 'morado_5may', 'Insulator-Defect Detection', 'Sheep Detection', 'Wood Defect Detection', 'Wind Turbines (by Kyle Graupe)', 'California and Arizona Wind Turbines (by Duke Dataplus2020)', 'Overhead Imagery of Wind Turbines (by Duke Dataplus2020)', 'Large Wind Turbines (by Duke Dataplus2020)', 'Wind Turbine Detection (by Saurabh Shahane)', 'Multi-topography Dataset for Wind Turbine Detection', 'YOLO Annotated Wind Turbine Surface Damage', 'Wind Turbine Detection (by Noah Vriese)', 'Self Driving Cars', 'SARAS-ESAD 2020', 'BCCD', 'Airbus Aircraft Detection', 'Airbus Oil Storage Detection', 'Food Recognition 2022', 'AFO', 'DeepGlobe Land Cover 2018', 'Safety Helmet Detection', 'Face Mask Detection', 'Road Sign Detection', 'Ship Detection from Aerial Images', 'VSAI', 'Semantic Segmentation Satellite Imagery', 'Malaria Bounding Boxes', 'NPU-BOLT', 'Fluorescent Neuronal Cells', 'Heat Sink Surface Defect', 'Bee Image Object Detection', 'UAV Small Object Detection (UAVOD-10)', 'Accurate Drone Shapes', 'Skin Cancer (HAM10000)', 'Accurate Nevus Shapes', 'iSAID Airplane Grayscale', 'AgRobTomato', 'RpiTomato', 'Vehicle Wheel Detection', 'TTPLA', 'Plastic Bottles', 'Industrial Optical Inspection', 'UAVOD-10', 'Cracks and Potholes in Road', 'Roundabout Aerial Images', 'Malaria Segmentation', 'Construction Vehicle Images', 'Teeth Segmentation on Dental X-ray Images', 'Breast Ultrasound Images', 'Fabric Stain', 'Road Pothole Images', 'Aerial Power Infrastructure', 'Detection of Small Size Construction Tools', 'Piling Sheet Image Data', 'MVTec AD', 'MVTec D2S', 'QuinceSet', 'MVTec LOCO AD', 'FloodNet 2021 (Track 1)', 'Supervisely Synthetic Crack Segmentation', 'Concrete Crack Segmentation', 'CNAM-CD', 'Photometric Stereo Leafs', 'CCAgT', 'WIDER FACE', 'OpenCow2020', 'UAVid', 'Automatic Monitoring of Pigs', 'Labeled Surgical Tools and Images', 'xView 2018', 'Fruit Recognition', 'Intraretinal Cystoid Fluid', 'Severstal', 'Maize Whole Plant Image', 'COCO-Stuff 10k', 'AeroScapes', 'EDD2020', 'PCBSegClassNet', 'CaT', 'Perrenial Plants Detection', 'Strawberry dataset for object detection', 'Panoramic Dental X-rays', 'AI4Agriculture Grape Dataset', 'Corn Leaf Infection', 'Crop and Weed Detection', 'TBX11K', 'Dentalai', 'Fetal Head UltraSound', 'DeepSeedling', 'Maize Tassel Detection', 'GWHD 2021', 'Cattle Detection and Counting in UAV Images', 'CBIS-DDSM', 'Substation Equipment', 'PolypGen', 'CelebAMask-HQ', 'Goat Image', 'OPPD', '38-Cloud', 'Sugar Beets 2016', 'MangoNet', 'Apple Dataset Benchmark from Orchard Environment', 'COCO 2017', 'CosegPP', 'EWS', 'COCO-Stuff 164k', 'Cows2021', 'BBBC041Seg', 'Multispectral Potato Plants Images', 'PLD-UAV', 'Rust and Leaf Miner in Coffee Crop', 'Strawberry Disease Detection', 'Supervisely Persons', 'Tree Species Detection', 'DeepGlobe 2018 Road Extraction', 'Annotated Quantitative Phase Microscopy Cell', 'CIHP', 'Plant Detection and Counting', 'PV', 'LVIS', "GlaS@MICCAI'2015: Gland Segmentation", 'ADE20K', 'MoDES-Cattle', 'PRMI', 'Indoor Objects Detection', 'EMPS', 'WaterDataset', 'Dhaka-AI', 'Lighter detection under x-ray', 'Alfalfa Roots', 'MSD', 'IITM-HeTra', 'Paddy Rice Imagery', 'Kvasir Instrument', 'SUIM', 'Danish Golf Courses Orthophotos', 'CCP', 'LADD', 'Makassar Road', 'Mini Traffic Detection', 'HRSC2016-MS', 'BTAD', 'MaskNet', 'SIIM-ACR Pneumothorax Segmentation 2019', 'Car Segmentation', 'DOTA', 'iSAID', 'Surgical Scene Segmentation in Robotic Gastrectomy', 'PACO-LVIS', 'Indian Roads', 'CitySegmentation', 'Pothole Detection', 'Mapillary Vistas', 'GTSDB', 'ItalianSigns', 'HIT-UAV', 'Strawberry Dataset for Object Detection', 'Urban Street: Flower Classification', 'Urban Street: Fruit Classification', 'Urban Street: Tree Classification', 'Urban Street: Leaf Classification', 'Urban Street: Branch', 'Urban Street: Leaf', 'Urban Street: Trunk', 'Urban Street: Tree', 'ABU Robocon 2021 Pot', 'Vehicle Dataset for YOLO', 'SISVSE', 'Outdoor Hazard Detection', 'HyperKvasir Images', 'OD-WeaponDetection: Knife Classification', 'OD-WeaponDetection: Pistol Classification', 'OD-WeaponDetection: Knife Detection', 'OD-WeaponDetection: Pistol Detection', 'OD-WeaponDetection: Sohas Classification', 'OD-WeaponDetection: Sohas Detection', 'AFID', 'Safety Helmet and Reflective Jacket', 'Сonstruction Equipment', 'Weapons in Images', 'Drone Dataset (UAV)', 'Tunisian Licensed Plates', 'Intruder Detection', 'Guns in an Active State Detection', 'Gesture v1.0', 'BLPR: License Plate Localization', 'BLPR: Character Recognition', 'Carrot-Weed', 'CADI-AI', 'Dataset of Annotated Food Crops and Weed Images', 'TACO', 'ROAD-SEC', 'AgRobTomato Dataset', 'RpiTomato Dataset', 'Pink-Eggs Dataset V1', 'Total-Text', 'Supervisely HRDA Plants Demo', 'Mini-Orchards', 'Simulated-Orchards', 'FPIC-Component', 'WeedMaize', 'LADD: Lacmus Drone Dataset', 'Non-Metal Lighter Target Detection Under X-Ray', 'Bangladeshi License Plate Recognition: License Plate Localization', 'Bangladeshi License Plate Recognition: Character Recognition', 'Maize-Weed Image', 'Google Recaptcha Image', 'SignverOD', 'Skin Cancer: HAM10000', 'Captcha Object Detection', 'FloodNet 2021: Track 1']

class NinjaDataset:
    def __init__(self, ds_path, split='train', flip_mode=False, special_ds_dir=None):
        assert split.lower() in ('train', 'test')
        self.flip_mode = flip_mode

        # Helper function to find case-insensitive match for directory names
        def find_dir(path, dir_name):
            for name in os.listdir(path):
                if name.lower() == dir_name:
                    return name
            return None

        # try to load the split using different names, this allivates inconsistent dataset naming
        train_dir = find_dir(ds_path, 'train') or find_dir(ds_path, 'labeled') or find_dir(ds_path, 'train_data') or find_dir(ds_path, 'train_val') or find_dir(ds_path, 'training')
        test_dir = find_dir(ds_path, 'test') or find_dir(ds_path, 'val') or find_dir(ds_path, 'validation') or find_dir(ds_path, 'test_data') or find_dir(ds_path, 'test_a') or find_dir(ds_path, 'testing')
        ds_dir = find_dir(ds_path, special_ds_dir or 'ds') or find_dir(ds_path, 'ds0')


        if train_dir and test_dir:
            self.has_splits = True
        elif ds_dir:
            self.has_splits = False
            all_imgs = sorted(os.listdir(os.path.join(ds_path, ds_dir, 'img')))
            all_anns = sorted(os.listdir(os.path.join(ds_path, ds_dir, 'ann')))
            indices = list(range(len(all_imgs)))
            random.seed(0)
            random.shuffle(indices)
            split_ratio = int(len(indices) * 0.8)
            if split.lower() == 'train':
                self.img_list = [os.path.join(ds_path, ds_dir, 'img', all_imgs[i]) for i in indices[:split_ratio]]
                self.ann_list = [os.path.join(ds_path, ds_dir, 'ann', all_anns[i]) for i in indices[:split_ratio]]
            elif split.lower() == 'test':
                self.img_list = [os.path.join(ds_path, ds_dir, 'img', all_imgs[i]) for i in indices[split_ratio:]]
                self.ann_list = [os.path.join(ds_path, ds_dir, 'ann', all_anns[i]) for i in indices[split_ratio:]]
            else:
                raise ValueError('Split must be either train or test')
        else:
            raise ValueError('Dataset must have either train and test splits or a ds folder')

        if self.has_splits:
            split_dir = test_dir if split.lower() == 'test' else train_dir
            if split_dir is None:
                raise ValueError(f"No directory found for split '{split}'")
            self.img_list = [os.path.join(ds_path, split_dir, 'img', fname) for fname in sorted(os.listdir(os.path.join(ds_path, split_dir, 'img')))]
            self.ann_list = [os.path.join(ds_path, split_dir, 'ann', fname) for fname in sorted(os.listdir(os.path.join(ds_path, split_dir, 'ann')))]

        # check correctness
        assert len(self.img_list) == len(self.ann_list) and Path(self.img_list[-1]).name.split('.')[0] == Path(self.ann_list[-1]).name.split('.')[0]


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # load paths
        imgpath, annpath = self.img_list[idx], self.ann_list[idx]
        # get image
        img = Image.open(imgpath)
        # get annotation
        with open(annpath, 'r') as f:
            ann = json.load(f)
        objects = [obj for obj in ann['objects'] if 'bitmap' in obj]
        
        # load segmentation masks, origin and class names
        rotated = False  # we will check if the image is rotated w.r.t. the masks
        segs, origins = [], []  # masks are given as a mask starting from origin
        class_names = [obj['classTitle'].lower().replace(' ', '-') for obj in objects]
        for mind, obj in enumerate(objects):
            seg = self.base64_2_data(obj['bitmap']['data'])
            origin = obj['bitmap']['origin']
            segs.append(seg)
            origins.append(origin)
            if origin[1] + seg.shape[0] > img.height or origin[0] + seg.shape[1] > img.width:
                rotated = True
        if rotated:
            for seg, origin in zip(segs, origins):
                assert origin[1] + seg.shape[0] <= img.width or origin[0] + seg.shape[1] <= img.height
            img = img.rotate(-90, expand=True)

        masks = np.zeros((len(objects), img.height, img.width), dtype=bool)
        for mind, (seg, origin) in enumerate(zip(segs, origins)):
            masks[mind, origin[1]:origin[1]+seg.shape[0], origin[0]:origin[0]+seg.shape[1]] = seg
        if self.flip_mode == 'central':
            masks = masks[:, ::-1, ::-1]  # flip masks vertically and horizontally, for some reason they are flipped in the dataset
        return img, masks, class_names
        
    @staticmethod
    def base64_2_data(s: str) -> np.ndarray:
        """
        Convert base64 encoded string to numpy array.

        :param s: Input base64 encoded string.
        :type s: str
        :return: Bool numpy array
        :rtype: :class:`np.ndarray`
        """
        z = zlib.decompress(base64.b64decode(s))
        n = np.frombuffer(z, np.uint8)

        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
        if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
            mask = imdecoded[:, :, 3].astype(bool)  # 4-channel imgs
        elif len(imdecoded.shape) == 2:
            mask = imdecoded.astype(bool)  # flat 2d mask
        else:
            raise RuntimeError("Wrong internal mask format.")
        return mask

def describe_sample(image, masks, class_names):
    description = {}
    description['class_names'] = class_names
    description['n_instances'] = len(masks)
    description['empty_mask'] = len(masks) == 0
    description['image_size'] = image.height * image.width
    description['object_size'] = [mask.sum() for mask in masks]
    description['canny_per'] = (cv2.Canny(np.array(image), 100, 200)>0).sum() / description['image_size'] 
    # skimage region properties
    labels = np.zeros((image.height, image.width), dtype=int)
    for mind, mask in enumerate(masks):
        labels[mask] = mind + 1
    regprops = skimage.measure.regionprops(labels)
    description['solidity'] = [regprop.solidity for regprop in regprops]
    description['euler_number'] = [regprop.euler_number for regprop in regprops]
    description['area_perimeter_ratio'] = [regprop.area / regprop.perimeter for regprop in regprops]
    return description



def describe_dataset(sample_descriptions, mask_avg):
    # get class names as the union of all the class names
    class_names = set()
    for sample_desc in sample_descriptions:
        class_names.update(sample_desc['class_names'])
    class_names = sorted(list(class_names))
    # get the number of instances per image
    n_instances_per_image = sum([sample_desc['n_instances'] for sample_desc in sample_descriptions]) / len(sample_descriptions)
    # get the percentage of empty masks
    empty_masks_per = sum([sample_desc['empty_mask'] for sample_desc in sample_descriptions]) / len(sample_descriptions)

    max_instances, min_instances = max([sample_desc['n_instances'] for sample_desc in sample_descriptions]), min([sample_desc['n_instances'] for sample_desc in sample_descriptions])
    # get flattened descriptions
    if max_instances > 1 or min_instances == 0:
        fdesc = []  # flattened list with one entry per dict
        for sample_desc in sample_descriptions:  # dicts with keys and lists as values
            for ind in range(len(sample_desc['class_names'])):
                if sample_desc['empty_mask']:
                    continue
                new_entry = {key: sample_desc[key][ind] if isinstance(sample_desc[key], list) else sample_desc[key] for key in sample_desc.keys()}
                fdesc.append(new_entry)
    else:
        fdesc = sample_descriptions

    # get the average object size
    obj_sizes = [desc['object_size'] for desc in fdesc]
    object_size_mean = np.array(obj_sizes).mean(axis=0)
    object_size_std = np.array(obj_sizes).std(axis=0)
    # get the average image size
    image_size_mean = np.array([sample_desc['image_size'] for sample_desc in sample_descriptions]).mean()
    image_size_std = np.array([sample_desc['image_size'] for sample_desc in sample_descriptions]).std()
    # get the average canny percentage
    canny_per_mean = np.array([sample_desc['canny_per'] for sample_desc in sample_descriptions]).mean()
    canny_per_std = np.array([sample_desc['canny_per'] for sample_desc in sample_descriptions]).std()
    # get the average solidity
    solidities = [desc['solidity'] for desc in fdesc]
    solidity_mean = np.array(solidities).mean(axis=0)
    solidity_std = np.array(solidities).std(axis=0)
    # get the average euler number
    euler_numbers = [desc['euler_number'] for desc in fdesc]
    euler_number_mean = np.array(euler_numbers).mean(axis=0)
    euler_number_std = np.array(euler_numbers).std(axis=0)
    # get the average area perimeter ratio
    area_perimeter_ratios = [desc['area_perimeter_ratio'] for desc in fdesc]
    area_perimeter_ratio_mean = np.array(area_perimeter_ratios).mean(axis=0)
    area_perimeter_ratio_std = np.array(area_perimeter_ratios).std(axis=0)
    # get the spatial unpredictability
    spatial_unpredictability = (-(mask_avg * np.log(mask_avg + 1e-8)).sum()) / (mask_avg.shape[0] * mask_avg.shape[1])
    # print the results
    retdict = {
        'single_class': len(class_names) == 1,
        'many_instances': n_instances_per_image > 5,
        'big_image': image_size_mean > 262144,
        'many_images': len(sample_descriptions) > 100,

        'n_classes': len(class_names),
        'n_instances_per_image': n_instances_per_image,
        'n_images': len(sample_descriptions),
        'empty_masks_per': empty_masks_per,
        'object_size_mean': object_size_mean,
        'object_size_std': object_size_std,
        'image_size_mean': image_size_mean,
        'image_size_std': image_size_std,
        'canny_per_mean': canny_per_mean,
        'canny_per_std': canny_per_std,
        'solidity_mean': solidity_mean,
        'solidity_std': solidity_std,
        'euler_number_mean': euler_number_mean,
        'euler_number_std': euler_number_std,
        'area_perimeter_ratio_mean': area_perimeter_ratio_mean,
        'area_perimeter_ratio_std': area_perimeter_ratio_std,
        'spatial_unpredictability': spatial_unpredictability,
    }
    print('-'*20)
    pprint(retdict)
    print('-'*20)

    return retdict

datasets = (
    'CHASE DB1',
    'Carrot-Weed',
    'MangoNet',
    'Supervisely HRDA Plants Demo',
    "GlaS@MICCAI'2015: Gland Segmentation",
    'Panoramic Dental X-rays',
    'WGISD',
    'Concrete Crack Segmentation',
    'EMPS',
    'Coffee Leaf Biotic Stress',
    'Alfalfa Roots',
    'Supervisely Synthetic Crack Segmentation',
    'MSD',
    'Water Meters',
    'SUIM',
    'Full Body TikTok Dancing',
    'KolektorSDD2',
    'Accurate Nevus Shapes',
    'AeroScapes',
    'FPIC-Component',
    'PolypGen',
    'Accurate Drone Shapes',
    'Synthetic Plants',
    'Skin Cancer (HAM10000)',
    'CIHP',
    )

def prepare_and_visualize():
    datasets_dir = '/home/franchesoni/adisk/ninjads/'
    N = 5
    for dataset in datasets:
        # get dst dir 
        ds_nickname = dataset.lower().replace(' ', '-')
        ds_path = os.path.join(datasets_dir, ds_nickname)
        dstdir = Path('gitignored') / ds_nickname
        shutil.rmtree(dstdir, ignore_errors=True)
        dstdir.mkdir(parents=True)
        # download the dataset if needed
        if not os.path.exists(ds_path):
            new_ds_path = dstools.download(dataset, dst_dir=datasets_dir)
            assert new_ds_path == ds_path
        # load the dataset
        flip_mode = 'central' if dataset in ('Carrot-Weed') else False
        special_ds_dir = 'segmentation1' if dataset == 'Panoramic Dental X-rays' else 'synthetic cracks' if dataset == 'Supervisely Synthetic Crack Segmentation' else None
        ds = NinjaDataset(ds_path, flip_mode=flip_mode, special_ds_dir=special_ds_dir)
        for ind, sample in tqdm.tqdm(enumerate(ds), total=N):  # for the first few samples save the images and masks
            if ind == N:
                break
            img, masks, class_names = sample
            if masks.shape[0] == 0:  # if masks is empty, check for all the dataset how many empty masks do we have
                with_mask, wo_mask = [], []
                count = 0
                for i, (img, masks, class_names) in tqdm.tqdm(enumerate(ds), total=len(ds)):
                    if masks.shape[0] > 0:
                        with_mask.append(i)
                        if count < N:
                            # save images and first N masks
                            img.save(dstdir / f'img_{i}.png')
                            for mind, mask in enumerate(masks[:N]):
                                Image.fromarray(mask).save(dstdir / f'mask_{i}_{mind}_{class_names[mind]}.png')
                            count += 1
                    else:
                        wo_mask.append(i)
                print('!'*20)
                print(f'{dataset} has {len(with_mask)} images with masks and {len(wo_mask)} without masks')
                print('!'*20)
                break
            # save images and first N masks
            img.save(dstdir / f'img_{ind}.png')
            for mind, mask in enumerate(masks[:N]):
                Image.fromarray(mask).save(dstdir / f'mask_{ind}_{mind}_{class_names[mind]}.png')

def describe_all_datasets():
    datasets_dir = '/home/franchesoni/adisk/ninjads/'
    for dataset in datasets:
        # get dst dir 
        ds_nickname = dataset.lower().replace(' ', '-')
        ds_path = os.path.join(datasets_dir, ds_nickname)
        assert os.path.exists(ds_path), 'dataset should exist, run prepare'
        # load the dataset
        flip_mode = 'central' if dataset in ('Carrot-Weed') else False
        special_ds_dir = 'segmentation1' if dataset == 'Panoramic Dental X-rays' else 'synthetic cracks' if dataset == 'Supervisely Synthetic Crack Segmentation' else None
        ds = NinjaDataset(ds_path, flip_mode=flip_mode, special_ds_dir=special_ds_dir)
        print(ds_nickname, len(ds))
        avg_mask_num = np.zeros((512,512), dtype=int)
        avg_mask_denom = 0
        descriptions = []
        for ind, sample in tqdm.tqdm(enumerate(ds), total=len(ds)):
            img, masks, class_names = sample
            sample_desc = describe_sample(img, masks, class_names)
            descriptions.append(sample_desc)
            avg_mask_num = avg_mask_num + skimage.transform.resize((masks*1).sum(axis=0), (512,512))
            avg_mask_denom += len(masks) 
            # if ind == 5:  # dev
            #     break
        avg_mask = avg_mask_num / avg_mask_denom
        ds_desc = describe_dataset(descriptions, avg_mask)
        np.save(f'gitignored/{ds_nickname}_desc.npy', ds_desc)
        






if __name__ == '__main__':
    # prepare_and_visualize()
    describe_all_datasets()
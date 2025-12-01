import os

import pickle

import json

import torch

from torchvision.transforms import Compose

from PIL import Image

import numpy as np

import random

from torch.utils.data.distributed import DistributedSampler # [DDP]



from config import cfg

from log import logger





class Cholec_Val(torch.utils.data.Dataset):

    def __init__(self,sp,transform=None,f=False,class_num: int = -1):

        self.data_path = "/data/cholecdata"

        self.transform = transform

        self.class_num = class_num

        self.is_sp = sp

        self.f = f

        data = self.read_data()

        self.data = data

    def read_data(self):

        data = []

        cholec80_frames = os.path.join(self.data_path,"cholec80/frames/val")

        cholec80_labels = os.path.join(self.data_path,"cholec80/labels/val/frame_phase_val.pkl")

        a = pickle.load(open(cholec80_labels,"rb"))

        invalid_80 = [42]

        for video in a.keys():

            id = int(video[5:])

            if id in invalid_80: continue

            video_folder = os.path.join(cholec80_frames,video)

            input = [(p["Frame_id"],p["Phase_gt"]) for p in a[video]]

            video_id = f"cholec80_{video}"

            for image,gt in input:

                filename = f"{image}.png"

                impath = os.path.join(video_folder,filename)

                label = torch.zeros(self.class_num)

                label[gt] = 1

                data.append([impath,label,video_id])

        def read_json(file_path):

            with open(file_path, 'r') as file: data = json.load(file)

            return data

        endo_path = os.path.join(self.data_path,"endoscapes/val")

        b = read_json(os.path.join(endo_path,"annotation_ds_coco.json"))

        for p in b['images']:

            filename,gt,video = (p['file_name'],(p['ds']),p['video_id'])

            impath = os.path.join(endo_path,filename)

            gt = [round(value) for value in gt]

            label = torch.zeros(self.class_num)

            for i in range(len(gt)):

                if gt[i] == 1: label[i+7] = 1

            video_id = f"endoscapes_{video}"

            if not self.f:

                if self.is_sp: label = self.random_pick_one(label)

            data.append([impath,label,video_id])

        val_split = [8,12,29,50,78]

        cholect50_frames = os.path.join(self.data_path,"cholect50/videos")

        cholect50_labels = os.path.join(self.data_path,"cholect50/labels")

        def get_label(labels):

            output = torch.zeros(self.class_num)

            for label in labels:

                index = label[0]

                if index == -1: continue

                output[index+10] = 1

            return output

        for i in val_split:

            video_folder = os.path.join(cholect50_frames,f"VID{i:02d}" if i < 100 else f"VID{i:03d}")

            label_file = os.path.join(cholect50_labels,f"VID{i:02d}.json" if i < 100 else f"VID{i:03d}.json")

            video_id = f"cholect50_{i}"

            with open(label_file, 'r') as file: a = json.load(file)

            for frame_id,gts in a['annotations'].items():

                filename = f"{int(frame_id):06d}.png"

                impath = os.path.join(video_folder,filename)

                label = get_label(gts)

                label = torch.Tensor(label)

                if not self.f:

                    if self.is_sp: label = self.random_pick_one(label)

                data.append([impath,label,video_id])

        return data

    def random_pick_one(self,tensor):

        ones_indices = torch.nonzero(tensor, as_tuple=False)

        if ones_indices.shape[0] > 0:

            random_idx = torch.randint((ones_indices.shape[0]), (1,))

            selected_index = ones_indices[random_idx]

            tensor.zero_()

            tensor[selected_index] = 1

        return tensor

    def __getitem__(self,index):

        impath, label, vid = self.data[index]

        image = Image.open(impath).convert("RGB")

        if self.f:

            if self.is_sp: label = self.random_pick_one(label)

        if self.transform: image = self.transform(image)

        if not self.f:

            if self.is_sp: assert torch.sum(label) < 2

        return image, label, vid

    def __len__(self): return len(self.data)

    def labels(self):

        with open('cholec/cholec_labels.txt', 'r')as f: text = f.read()

        return text.split('\n')



class Cholec_Train(torch.utils.data.Dataset):

    def __init__(self,transform=None,f=False,partial=False,class_num: int = -1):

        self.data_path = "/data/cholecdata"

        self.transform = transform

        self.class_num = class_num

        self.f = f

        self.partial = partial

        self.data = self.read_data()

        self. original_labels = self._copy_original_labels()
    def read_data(self):

        data = []

        cholec80_frames = os.path.join(self.data_path,"cholec80/frames/train")

        cholec80_labels = os.path.join(self.data_path,"cholec80/labels/train/frame_phase_train.pkl")

        a = pickle.load(open(cholec80_labels,"rb"))

        invalid_80 = [6,10,14,32]

        for video in a.keys():

            id = int(video[5:])

            if id in invalid_80: continue

            video_folder = os.path.join(cholec80_frames,video)

            input = [(p["Frame_id"],p["Phase_gt"]) for p in a[video]]

            video_id = f"cholec80_{video}"

            for image,gt in input:

                filename = f"{image}.png"

                impath = os.path.join(video_folder,filename)

                label = torch.zeros(self.class_num)

                label[gt] = 1

                data.append([impath,label,video_id])

        def read_json(file_path):

            with open(file_path, 'r') as file: data = json.load(file)

            return data

        endo_path = os.path.join(self.data_path,"endoscapes/train")

        b = read_json(os.path.join(endo_path,"annotation_ds_coco.json"))

        for p in b['images']:

            filename,gt,video = (p['file_name'],(p['ds']),p['video_id'])

            impath = os.path.join(endo_path,filename)

            gt = [round(value) for value in gt]

            label = torch.zeros(self.class_num)

            for i in range(len(gt)):

                if gt[i] == 1: label[i+7] = 1

            video_id = f"endoscapes_{video}"

            if not self.partial:

                if not self.f: label = self.random_pick_one(label)

            data.append([impath,label,video_id])

        train_split = [1,2,4,5,13,15,18,22,23,25,26,27,31,35,36,40,43,47,48,49,52,56,57,60,62,65,66,68,70,75,79,92,96,103,110]

        cholect50_frames = os.path.join(self.data_path,"cholect50/videos")

        cholect50_labels = os.path.join(self.data_path,"cholect50/labels")

        def get_label(labels):

            output = torch.zeros(self.class_num)

            for label in labels:

                index = label[0]

                if index == -1: continue

                output[index+10] = 1

            return output

        for i in train_split:

            video_folder = os.path.join(cholect50_frames,f"VID{i:02d}" if i < 100 else f"VID{i:03d}")

            label_file = os.path.join(cholect50_labels,f"VID{i:02d}.json" if i < 100 else f"VID{i:03d}.json")

            video_id = f"cholect50_{i}"

            with open(label_file, 'r') as file: a = json.load(file)

            for frame_id,gts in a['annotations'].items():

                filename = f"{int(frame_id):06d}.png"

                impath = os.path.join(video_folder,filename)

                label = get_label(gts)

                if not self.partial:

                    if not self.f: label = self.random_pick_one(label)

                data.append([impath,label,video_id])

        assert len(set([d[0] for d in data])) == len([d[1] for d in data]) == len(data)

        return data
    
    def _copy_original_labels(self):
        """保存原始标签的副本"""
        original = []
        for entry in self.data:
            label = entry[1]
            if isinstance(label, torch. Tensor):
                original.append(label.clone())
            else:
                original.append(torch.tensor(label). clone())
        return original
    
    def update_labels(self, pseudo_labels, indices):
        """
        更新伪标签（每个 epoch 重新生成，基于原始标签）
        
        最终标签 = 原始标签 OR 伪标签
        """
        if len(pseudo_labels) != len(indices):
            logger.warning(f"[CAP] Mismatch: {len(pseudo_labels)} labels vs {len(indices)} indices")
            return
        
        count_changed = 0
        
        for i, global_idx in enumerate(indices):
            idx = int(global_idx)
            
            if idx < 0 or idx >= len(self.data):
                continue
            
            # 获取原始标签（不是当前可能已被 CAP 修改的标签）
            orig_label = self. original_labels[idx]
            if isinstance(orig_label, torch.Tensor):
                np_orig = orig_label. numpy(). copy()
            else:
                np_orig = np.array(orig_label). copy()
            
            # 最终标签 = 原始正样本 OR 伪正标签
            new_label = np. maximum(np_orig, pseudo_labels[i])
            
            # 检查是否有变化
            current = self.data[idx][1]
            if isinstance(current, torch. Tensor):
                current_np = current.numpy()
            else:
                current_np = np.array(current)
            
            if not np.array_equal(current_np, new_label):
                count_changed += 1
                self.data[idx][1] = torch.from_numpy(new_label). float()
        
        logger.info(f"[CAP] Updated {count_changed} labels out of {len(indices)} samples")

    def random_pick_one(self,tensor):

        ones_indices = torch.nonzero(tensor, as_tuple=False)

        if ones_indices.shape[0] > 0:

            random_idx = torch.randint((ones_indices.shape[0]), (1,))

            selected_index = ones_indices[random_idx]

            tensor.zero_()

            tensor[selected_index] = 1

        return tensor

    def __getitem__(self, index):
        impath, label, vid = self.data[index]
        image = Image.open(impath).convert("RGB")

        if not self.partial:
            if self.f: label = self.random_pick_one(label)

        if self.transform: image = self.transform(image)
        if not self.partial: assert torch.sum(label) < 2

        # [修改] 返回 4 个值，最后加上 index
        return image, label, vid, index

    def __len__(self): return len(self.data)

    def labels(self):

        with open('cholec/cholec_labels.txt', 'r')as f: text = f.read()

        return text.split('\n')

    

class Cholec_Test(torch.utils.data.Dataset):

    def __init__(self,transform=None,class_num: int = -1):

        self.data_path = "/data/cholecdata"

        self.transform = transform

        self.class_num = class_num

        self.data = self.read_data()

    def read_data(self):

        data = []

        cholec80_frames = os.path.join(self.data_path,"cholec80/frames/test")

        cholec80_labels = os.path.join(self.data_path,"cholec80/labels/test/frame_phase_test.pkl")

        a = pickle.load(open(cholec80_labels,"rb"))

        valid_80 = [51, 53, 54, 55, 58, 59, 61, 63, 64, 69, 73, 74, 76, 77, 80]

        for video in a.keys():

            id = int(video[5:])

            if id not in valid_80: continue

            video_folder = os.path.join(cholec80_frames,video)

            input = [(p["Frame_id"],p["Phase_gt"]) for p in a[video]]

            video_id = f"cholec80_{video}"

            for image,gt in input:

                filename = f"{image}.png"

                impath = os.path.join(video_folder,filename)

                label = torch.zeros(self.class_num)

                label[gt] = 1

                data.append([impath,label,video_id])

        def read_json(file_path):

            with open(file_path, 'r') as file: data = json.load(file)

            return data

        endo_path = os.path.join(self.data_path,"endoscapes/test")

        b = read_json(os.path.join(endo_path,"annotation_ds_coco.json"))

        for p in b['images']:

            filename,gt,video = (p['file_name'],(p['ds']),p['video_id'])

            impath = os.path.join(endo_path,filename)

            gt = [round(value) for value in gt]

            label = torch.zeros(self.class_num)

            for i in range(len(gt)):

                if gt[i] == 1: label[i+7] = 1

            video_id = f"endoscapes_{video}"

            data.append([impath,label,video_id])

        test_split = [6,10,14,32,42,51,73,74,80,111]

        cholect50_frames = os.path.join(self.data_path,"cholect50/videos")

        cholect50_labels = os.path.join(self.data_path,"cholect50/labels")

        def get_label(labels):

            output = torch.zeros(self.class_num)

            for label in labels:

                index = label[0]

                if index == -1: continue

                output[index+10] = 1

            return output

        for i in test_split:

            video_folder = os.path.join(cholect50_frames,f"VID{i:02d}" if i < 100 else f"VID{i:03d}")

            label_file = os.path.join(cholect50_labels,f"VID{i:02d}.json" if i < 100 else f"VID{i:03d}.json")

            video_id = f"cholect50_{i}"

            with open(label_file, 'r') as file: a = json.load(file)

            for frame_id,gts in a['annotations'].items():

                filename = f"{int(frame_id):06d}.png"

                impath = os.path.join(video_folder,filename)

                label = get_label(gts)

                data.append([impath,label,video_id])

        return data

    def __getitem__(self,index):

        impath, label, vid = self.data[index]

        image = Image.open(impath).convert("RGB")

        if self.transform: image = self.transform(image)

        return image , label, vid

    def __len__(self): return len(self.data)

    def labels(self):

        with open('cholec/cholec_labels.txt', 'r')as f: text = f.read()

        return text.split('\n')

    

class CholecDataset(torch.utils.data.Dataset):

    def __init__(self,label_list,transform=None,class_num: int = -1):

        self.transform = transform

        self.class_num = class_num

        self.data = label_list
        
    def update_labels(self, new_labels, indices):
        """
        new_labels: 当前 GPU 计算出的伪标签 [N_subset, C]
        indices: 这些伪标签对应的全局索引 [N_subset]
        """
        # 安全检查
        if len(new_labels) != len(indices):
            logger.warning(f"[CAP] Error: Labels ({len(new_labels)}) and Indices ({len(indices)}) mismatch!")
            return

        count_changed = 0
        
        # [核心逻辑] 根据索引，精准更新 self.data 中的特定行
        for i, global_idx in enumerate(indices):
            # 确保索引是整数
            idx = int(global_idx)
            
            # 获取原始数据：self.data[idx] = [impath, label, vid]
            original_entry = self.data[idx]
            original_label = original_entry[1]
            
            # 类型转换处理
            if isinstance(original_label, torch.Tensor):
                np_orig = original_label.numpy()
            else:
                np_orig = np.array(original_label)
            
            # 取并集：只填补假阴性 (0->1)，不修改已有正样本 (1->1)
            new_label_vec = new_labels[i]
            updated_label_np = np.maximum(np_orig, new_label_vec)
            
            # 如果有变化，则更新
            if not np.array_equal(np_orig, updated_label_np):
                count_changed += 1
                # 写回 Dataset
                self.data[idx][1] = torch.from_numpy(updated_label_np).float()
        
        logger.info(f"[CAP] Rank update: {len(new_labels)} samples processed, {count_changed} labels improved.")
        
    # 1. 修改 __getitem__，多返回一个 index
    def __getitem__(self, index):
        impath, label, vid = self.data[index]
        image = Image.open(impath).convert("RGB")

        if not self.partial:
            if self.f: label = self.random_pick_one(label)

        if self.transform: image = self.transform(image)
        if not self.partial: assert torch.sum(label) < 2

        # [修改] 返回 4 个值，最后加上 index
        return image, label, vid, index
    
    def __len__(self): return len(self.data)

    def labels(self):

        with open('cholec/cholec_labels.txt', 'r')as f: text = f.read()

        return text.split('\n')

    

class InitData(torch.utils.data.Dataset):

    def __init__(self,label_list,transform=None,class_num: int = -1):

        self.transform = transform

        self.class_num = class_num

        self.data = label_list

    def __getitem__(self,index):

        d = self.data[f"{index}"]

        impath = d['impath']

        label = d['label']

        image = Image.open(impath).convert("RGB")

        label = torch.Tensor(label)

        if self.transform: image = self.transform(image)

        return image , label

    def __len__(self): return len(self.data)

    def labels(self):

        with open('cholec/cholec_labels.txt', 'r')as f: text = f.read()

        return text.split('\n')


def build_cholec_dataset(train_preprocess: Compose,
                         val_preprocess: Compose,
                         pin_memory=True,
                         distributed=False): # [DDP] Added arg
    
    # ----------------------------------------------------
    # 1. Load Datasets based on Config
    # ----------------------------------------------------
    if cfg.use_lfile:
        logger.info(f'Using label file....{cfg.label_file}')
        label_file = cfg.label_file
        with open(label_file, 'r') as a:
            f = json.load(a)
        
        datasets = {}
        for key in f.keys():
            if key not in datasets: datasets[key] = []
            for ki in f.get(key):
                datasets[key].extend([[k, v.get('label'), v.get('videoID')] for k, v in f[key][ki].items() if os.path.exists(k)])

        if cfg.val_sp:
            logger.info("Loading single positive validation")
            val_sp_dataset = CholecDataset(datasets['val_sp'], val_preprocess, class_num=cfg.num_classes)
        
        logger.info("Loading partial positive validation")
        val_dataset   = CholecDataset(datasets['val_full'], val_preprocess, class_num=cfg.num_classes)
        train_dataset = CholecDataset(datasets['train'], train_preprocess, class_num=cfg.num_classes)
        test_dataset  = CholecDataset(datasets['test'], val_preprocess, class_num=cfg.num_classes)

    else:
        if cfg.getitem: 
            logger.info('Using getitem....')
        else: 
            logger.info('Using init labels....')
            
        # [关键修复]: 修正参数传递，明确指定关键字参数，防止位置参数错乱
        # Cholec_Val 定义: __init__(self, sp, transform=None, f=False, class_num: int = -1)
        val_dataset = Cholec_Val(sp=False, transform=val_preprocess, f=cfg.getitem, class_num=cfg.num_classes)
        val_sp_dataset = Cholec_Val(sp=True, transform=val_preprocess, f=cfg.getitem, class_num=cfg.num_classes)
        
        # Cholec_Train 定义: __init__(self, transform=None, f=False, partial=False, class_num: int = -1)
        train_dataset = Cholec_Train(transform=train_preprocess, f=cfg.getitem, partial=cfg.partial, class_num=cfg.num_classes)
        
        # Cholec_Test 定义: __init__(self, transform=None, class_num: int = -1)
        # 之前的错误代码传递了多余的参数 (False, cfg.getitem)，导致参数错位
        test_dataset = Cholec_Test(transform=val_preprocess, class_num=cfg.num_classes)

    # ----------------------------------------------------
    # 2. Load Init Datasets (Optional)
    # ----------------------------------------------------
    if cfg.perform_init:
        init_train_file = cfg.init_train_file
        with open(init_train_file, 'r') as b:
            init_train_data = json.load(b)
        init_train_dataset = InitData(init_train_data, train_preprocess, class_num=cfg.num_classes)
        
        init_val_file = cfg.init_val_file
        with open(init_val_file, 'r') as c:
            init_val_data = json.load(c)
        init_val_dataset = InitData(init_val_data, val_preprocess, class_num=cfg.num_classes)

    # ----------------------------------------------------
    # 3. Logging Info
    # ----------------------------------------------------
    if cfg.perform_init:
        logger.info(f"Length of init train data : {len(init_train_dataset)}")
        logger.info(f"Length of init val data : {len(init_val_dataset)}")
    logger.info(f"Length of train : {len(train_dataset)}")
    logger.info(f"Length of val : {len(val_dataset)}")
    if cfg.val_sp:
        logger.info(f"Length of val sp : {len(val_sp_dataset)}")
    logger.info(f"Length of test : {len(test_dataset)}")

    # ----------------------------------------------------
    # 4. Setup Samplers and DataLoaders [DDP Fixed]
    # ----------------------------------------------------
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler   = None  # Master Only Validation
        test_sampler  = None  # Master Only Testing
        
        if cfg.val_sp:
            val_sp_sampler = None
        
        if cfg.perform_init:
            init_train_sampler = DistributedSampler(init_train_dataset)
            init_val_sampler   = None 
    else:
        train_sampler = None
        val_sampler   = None
        test_sampler  = None
        
        if cfg.val_sp:
            val_sp_sampler = None
            
        if cfg.perform_init:
            init_train_sampler = None
            init_val_sampler = None

    # --- Train Loader ---
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        sampler=train_sampler,
        persistent_workers=True
    )

    # --- Val Loader ---
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
        sampler=val_sampler,
        persistent_workers=True
    )
    
    # --- Val SP Loader ---
    val_sp_loader = None
    if cfg.val_sp:
        val_sp_loader = torch.utils.data.DataLoader(
            val_sp_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
            sampler=val_sp_sampler,
            persistent_workers=True
        )
        
    # --- Test Loader ---
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
        sampler=test_sampler,
        persistent_workers=True
    )
    
    # --- Init Loaders ---
    init_train_loader = None
    init_val_loader = None
    if cfg.perform_init:
        init_train_loader = torch.utils.data.DataLoader(
            init_train_dataset,
            batch_size=cfg.batch_size,
            shuffle=(init_train_sampler is None),
            num_workers=cfg.workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
            sampler=init_train_sampler,
            persistent_workers=True
        )
        
        init_val_loader = torch.utils.data.DataLoader(
            init_val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
            sampler=init_val_sampler,
            persistent_workers=True
        )

    logger.info("Build dataset done.")
    
    # 返回列表 (保持索引顺序)
    loaders = [train_loader, val_loader]
    if cfg.val_sp:
        loaders.append(val_sp_loader)
    
    loaders.append(test_loader)
    
    if cfg.perform_init:
        loaders.append(init_train_loader)
        loaders.append(init_val_loader)
        
    return loaders
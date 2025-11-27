import os

import torch
from torchvision.transforms import Compose
from torch.utils.data.distributed import DistributedSampler

from config import cfg
from log import logger
from utils import COCO_missing_dataset, COCO_missing_val_dataset, CocoDetection
from dataloader import build_cholec_dataset


def build_dataset(train_preprocess: Compose,
                  val_preprocess: Compose,
                  pin_memory=True,
                  distributed=False): # [修改] 增加 distributed 参数
    
    if "coco" in cfg.data:
        logger.info("Building coco dataset...")
        return build_coco_dataset(train_preprocess, val_preprocess, pin_memory, distributed=distributed)
    elif "nuswide" in cfg.data:
        logger.info("Building nuswide dataset...")
        return build_nuswide_dataset(train_preprocess, val_preprocess, pin_memory, distributed=distributed)
    elif "voc" in cfg.data:
        logger.info("Building voc dataset...")
        return build_voc_dataset(train_preprocess, val_preprocess, pin_memory, distributed=distributed)
    elif "cub" in cfg.data:
        logger.info("Building cub dataset...")
        return build_cub_dataset(train_preprocess, val_preprocess, pin_memory, distributed=distributed)
    elif "cholec" in cfg.data:
        logger.info("Building cholec dataset...")
        # [关键修改] 这里将 distributed 参数传给了 dataloader.py 中的 build_cholec_dataset
        return build_cholec_dataset(train_preprocess, val_preprocess, pin_memory, distributed=distributed)
    else:
        assert (False)


def build_coco_dataset(train_preprocess: Compose,
                       val_preprocess: Compose,
                       pin_memory=True,
                       distributed=False):
    # COCO Data loading
    instances_path_val = os.path.join(cfg.data,
                                      'annotations/instances_val2014.json')
    instances_path_train = cfg.dataset

    data_path_val = f'{cfg.data}/val2014'
    data_path_train = f'{cfg.data}/train2014'
    val_dataset = CocoDetection(data_path_val, instances_path_val,
                                val_preprocess)
    train_dataset = COCO_missing_dataset(data_path_train,
                                         instances_path_train,
                                         train_preprocess,
                                         class_num=cfg.num_classes)

    # DDP Sampler logic
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers,
        pin_memory=pin_memory,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False,
        sampler=val_sampler)

    logger.info("Build dataset done.")
    return [train_loader, val_loader]


def build_voc_dataset(train_preprocess: Compose,
                      val_preprocess: Compose,
                      pin_memory=True,
                      distributed=False):
    # VOC Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}VOC2012/JPEGImages'
    data_path_train = f'{cfg.data}VOC2012/JPEGImages'

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = COCO_missing_dataset(data_path_train,
                                         instances_path_train,
                                         train_preprocess,
                                         class_num=cfg.num_classes)
    
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg.workers,
                                               pin_memory=pin_memory,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False,
                                             sampler=val_sampler)
    logger.info("Build dataset done.")
    return [train_loader, val_loader]


def build_nuswide_dataset(train_preprocess: Compose,
                          val_preprocess: Compose,
                          pin_memory=True,
                          distributed=False):
    # Nus_wide Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}images'
    data_path_train = f'{cfg.data}images'

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = COCO_missing_dataset(data_path_train,
                                         instances_path_train,
                                         train_preprocess,
                                         class_num=cfg.num_classes)
    
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg.workers,
                                               pin_memory=pin_memory,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False,
                                             sampler=val_sampler)
    logger.info("Build dataset done.")
    return [train_loader, val_loader]

def build_cub_dataset(train_preprocess: Compose,
                          val_preprocess: Compose,
                          pin_memory=True,
                          distributed=False):
    # Nus_wide Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}CUB_200_2011/images'
    data_path_train = f'{cfg.data}CUB_200_2011/images'

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = COCO_missing_dataset(data_path_train,
                                         instances_path_train,
                                         train_preprocess,
                                         class_num=cfg.num_classes)
    
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg.workers,
                                               pin_memory=pin_memory,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False,
                                             sampler=val_sampler)
    logger.info("Build dataset done.")
    return [train_loader, val_loader]
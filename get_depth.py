from cmath import nan
import imp
import tqdm
import os, sys
from tkinter import image_names
from opt import config_parser
import torch
from collections import defaultdict
import random
from torch.utils.data import DataLoader
from datasets import dataset_dict
import pdb
# models
from models.nerf import *
from models.rendering import render_grid, render_rays, render_rays1, render_rays2, render_sh, render_sh_sample
from models.HashSiren import *
import numpy as np
# optimizer, scheduler, visualization, NeRV utils
from utils import *
import torch.optim as optim

# losses
from losses import loss_dict, MSELoss1, TVLoss_3
import imageio
# metrics
from metrics import *
from torchvision import transforms
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler.profilers import AdvancedProfiler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

if __name__ == '__main__':
    args = config_parser()
    size = [273, 273, 223]
    dataset = dataset_dict[args.dataset_name]
    # self.train_dataset = dataset(split='train', **kwargs)
    # self.val_dataset = dataset(split='val', **kwargs)
    train_dir = val_dir = args.root_dir
    # self.train_dataset = dataset(root_dir=train_dir, split='train', max_len=-1)
    # self.val_dataset   = dataset(root_dir=val_dir, split='val', max_len=10)
    train_dataset = dataset(root_dir=train_dir, split='train', img_wh = args.img_wh, depth = True)
    
    
    model_HashSiren = HashMlp1(hash_mod = True,
                hash_table_length = 273*273*223,
                in_features = 28, 
                hidden_features = 64, 
                hidden_layers = 2, 
                out_features = args.out_features,
                outermost_linear=True).to(device)
    if args.ckpt_path:
            print("loading model     ")
            ckpt = torch.load(args.ckpt_path)
            model_HashSiren.load_state_dict(ckpt['model_HashSiren'])


    output_feature = model_HashSiren(torch.tensor(0, device = device))
    output_feature = output_feature.reshape(size[2], size[1], size[0], 28).permute(3,0,1,2).float()
    sigama = output_feature[0:1]


    
    
            
    x = np.linspace(0, size[0]-1, size[0])
    y = np.linspace(0, size[1]-1, size[1])
    z = np.linspace(0, size[2]-1, size[2])
    xx = np.repeat(x[None, :], len(y), axis=0)  # 第一个None对应Z，第二个None对应Y；所以后面是(len(z), len(y))
    xxx = np.repeat(xx[None, :, :], len(z), axis=0)
    yy = np.repeat(y[:, None], len(x), axis=1)
    yyy = np.repeat(yy[None, :, :], len(z), axis=0)
    zz = np.repeat(z[:, None], len(y), axis=1)
    zzz = np.repeat(zz[:, :, None], len(x), axis=2)
    coors = np.concatenate((zzz[:, :, :, None], yyy[:, :, :, None], xxx[:, :, :, None]), axis=-1)
    sigama1 = sigama.permute(1,2,3,0).cpu().detach().numpy()
    coor = np.concatenate((coors,sigama1), axis=-1)

    coor = coor.reshape(-1,4)
    coor[:,3] = np.int64(coor[:,3]>0)
    index =np.array(coor[:,3], dtype=bool)
    coor1 = coor[index,:3]
    box_min = coor1.min(0)   #z y x
    box_max = coor1.max(0)
    box_min = torch.from_numpy(box_min[::-1].copy())
    box_max = torch.from_numpy(box_max[::-1].copy())


    xyz_min, xyz_max = train_dataset.get_box()
    coor_min = (box_min - torch.tensor((0,0,0)))/torch.tensor((size[0]-1,size[1]-1,size[2]-1))*(xyz_max - xyz_min) + xyz_min
    coor_max = (box_max - torch.tensor((0,0,0)))/torch.tensor((size[0]-1,size[1]-1,size[2]-1))*(xyz_max - xyz_min) + xyz_min
    grid_bounds = [coor_min.cuda(), coor_max.cuda()]        
    _set_grid_resolution_blender(args.num_voxels, coor_max.cuda(), coor_min.cuda())
    def decode_batch(batch):

        # rays = batch['rays'] # (B, 9)
        # rgbs = batch['rgbs'] # (B, 3)
        # image_t = batch['image_t']
        # view = batch['view']
        # pixel_choose = batch['pixel_choose']
        # image_t = batch['time']
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs
    def forward_render(rays, world_size, grid_bounds):
            import time
            # torch.cuda.synchronize()
            # start = time.time()
            """Do batched inference on rays using chunk."""
            B = rays.shape[0]  
            results = defaultdict(list)
            for i in range(0, B, args.chunk):
                rendered_ray_chunks = \
                    render_sh_sample(models,
                                embeddings,
                                rays[i:i+args.chunk],   #[32768, 8]
                                world_size,
                                grid_bounds,
                                args.N_samples,
                                args.use_disp,
                                args.perturb,
                                args.noise_std,
                                args.N_importance,
                                args.chunk, # chunk size is effective in val mode  32768
                                train_dataset.white_back, 
                                
                                test_time=False
                                )
                # torch.cuda.synchronize()
                # start1 = time.time()
                # print("forward2 :",start1-start)
                for k, v in rendered_ray_chunks.items():
                    results[k] += [v]   #k  'rgb_coarse'  v为数值
                # torch.cuda.synchronize()
                # start2 = time.time()
                # print("xunhuan  :",start2-start)
            # torch.cuda.synchronize()
            # start3 = time.time()
            for k, v in results.items():
                results[k] = torch.cat(v, 0)
            # torch.cuda.synchronize()
            # start4 = time.time()
            # print("forward3 :",start4-start3)
            return results
    
    train_dataset = DataLoader(dataset = train_dataset,
                            batch_size = 1024*2,
                            num_workers= 0,
                            shuffle=False)
    for i,sample in tqdm(enumerate(train_dataset)):
        rays, rgbs = decode_batch(sample)
        rays = rays.squeeze()
        rgbs = rgbs.squeeze()
        results = forward_render(rays, world_size, grid_bounds)
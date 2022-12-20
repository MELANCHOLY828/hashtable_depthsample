###不用MLP网络，使用球谐函数
from cmath import nan
import imp
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
if __name__ == '__main__':
    ckpt = "/data/zhangruiqi/lfy/data/results/hash_table/checkpoints/sample_again/ckpts/HashTable_2.tar"
    args = config_parser()
    world_size = torch.tensor([173, 289, 333])
    model_HashSiren = HashMlp(hash_mod = True,
                 hash_table_length = world_size.prod(),
                # hash_table_length = 333*184*272,   # x,y,z
                 in_features = args.in_features, 
                 hidden_features = args.hidden_features, 
                 hidden_layers = args.hidden_layers, 
                 out_features = args.out_features,
                 outermost_linear=True).cuda()
    ckpt = torch.load(ckpt)
    model_HashSiren.load_state_dict(ckpt['model_HashSiren'])
    import numpy as np
    # world_size = [173, 289, 333]
    x = np.linspace(0, world_size[0]-1, world_size[0])
    y = np.linspace(0, world_size[1]-1, world_size[1])
    z = np.linspace(0, world_size[2]-1, world_size[2])
    xx = np.repeat(x[None, :], len(y), axis=0)  # 第一个None对应Z，第二个None对应Y；所以后面是(len(z), len(y))
    xxx = np.repeat(xx[None, :, :], len(z), axis=0)
    yy = np.repeat(y[:, None], len(x), axis=1)
    yyy = np.repeat(yy[None, :, :], len(z), axis=0)
    zz = np.repeat(z[:, None], len(y), axis=1)
    zzz = np.repeat(zz[:, :, None], len(x), axis=2)
    coors = np.concatenate((zzz[:, :, :, None], yyy[:, :, :, None], xxx[:, :, :, None]), axis=-1)   # 这里zzz, yyy, xxx的顺序别错了，不然不好理解

    # X, Y, Z = np.meshgrid(x, y, z)
    # coors = np.concatenate((X[:, :, :, None], Y[:, :, :, None], Z[:, :, :, None]), axis=-1)
    output_feature = model_HashSiren(torch.tensor(0).cuda())


    output_feature = output_feature.reshape(world_size[2], world_size[1], world_size[0], 28).permute(3,0,1,2).float()
    sigama = output_feature[0:1]
    sigama1 = sigama.permute(1,2,3,0).cpu().detach().numpy()
    # sigama1 = sigama.permute(3,2,1,0).cpu().detach().numpy()
    coor = np.concatenate((coors,sigama1), axis=-1)

    coor = coor.reshape(-1,4)
    np.savetxt('/data/zhangruiqi/lfy/data/sample_again.txt',coor)
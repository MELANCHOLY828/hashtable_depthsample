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
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class NeRFSystem(LightningModule):    
    def __init__(self, args):
        super(NeRFSystem, self).__init__()
        # self.automatic_optimization=False       #把自动优化关掉
        self.back0 = 0
        self.args = args
        self.idx = 0
        self.idx_gpus = -1
        self.loss = loss_dict[args.loss_type]()
        
        self.save_path = '/data/liufengyi/Results'
        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]
        self.list_log = []
        self.img_wh = self.args.img_wh
        self.image_shape = torch.zeros(self.img_wh[1], self.img_wh[0], 3)

        self.tvloss_sh = TVLoss_3(self.args.TVLoss_weight_SH)
        self.tvloss_sigma = TVLoss_3(self.args.TVLoss_weight_sigama)
        
        # self.img_pixel = list(range(0, 480 * 640))
        # random.shuffle(self.img_pixel)

        
        self.size = [273, 273, 223]
        self.model_HashSiren = HashMlp1(hash_mod = True,
                 hash_table_length = 273*273*223,
                 in_features = 28, 
                 hidden_features = 64, 
                 hidden_layers = 2, 
                 out_features = self.args.out_features,
                 outermost_linear=True).cuda()
        
        # self.VoxelGrid = VoxelGrid(hash_mod = True,
        #          hash_table_length = 171*171*139,
        #          in_features = self.args.in_features, 
        #          hidden_features = self.args.hidden_features, 
        #          hidden_layers = self.args.hidden_layers, 
        #          out_features = self.args.out_features,
        #          outermost_linear=True).cuda()
        
        if self.args.ckpt_path:
            print("loading model     ")
            ckpt = torch.load(self.args.ckpt_path)
            self.model_HashSiren.load_state_dict(ckpt['model_HashSiren'])
            # table = ckpt['model_HashSiren']['table']
            # self.model_HashSiren.table.data = table    #加载了HashTable的值
            # self.model_HashSiren.table.requires_grad = False

        #改变box范围
        output_feature = self.model_HashSiren(torch.tensor(0, device = device))
        output_feature = output_feature.reshape(self.size[2], self.size[1], self.size[0], 28).permute(3,0,1,2).float()
        sigama = output_feature[0:1]


        
        
                
        x = np.linspace(0, self.size[0]-1, self.size[0])
        y = np.linspace(0, self.size[1]-1, self.size[1])
        z = np.linspace(0, self.size[2]-1, self.size[2])
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
        self.box_min = coor1.min(0)   #z y x
        self.box_max = coor1.max(0)
        self.box_min = torch.from_numpy(self.box_min[::-1].copy())
        self.box_max = torch.from_numpy(self.box_max[::-1].copy())
        # self.box_min = torch.from_numpy(self.box_min.copy())
        # self.box_max = torch.from_numpy(self.box_max.copy())
        self.models = [self.model_HashSiren]
        # self.models = [self.model_HashSiren]
        # self.model_MLP_dir = MLP_dir().cuda()
        # if self.args.ckpt_path:
        #     self.model_MLP_dir.load_state_dict(ckpt['model_MLP_dir'])
        #     for i in self.model_MLP_dir.parameters():
        #         i.requires_grad = False
        # self.models += [self.model_MLP_dir]
        
        # self.nerf_coarse = NeRF()
        # load_ckpt(self.nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    
        # # self.nerf_coarse.load_state_dict(ckpt_nerf['nerf_coarse_state_dict'])
        # self.models = [self.nerf_coarse]
        # if args.N_importance > 0:
        #     self.nerf_fine = NeRF()
        #     load_ckpt(self.nerf_fine, args.ckpt_path, model_name='nerf_fine')
        #     # self.nerf_fine.load_state_dict(ckpt_nerf['nerf_fine_state_dict'])
        #     self.models += [self.nerf_fine]
        # self.LatentCode = LatentCode()
        # load_ckpt(self.LatentCode, args.ckpt_path, model_name='LatentCode')
        # # self.LatentCode.load_state_dict(ckpt_nerf['latent_code'])
        # self.models += [self.LatentCode]
        # self.t_normalize = 0
        # self.flag_image_num = 0
        self.flag_epoch1 = 0
        # self.flag_epoch2 = 0

        # self.list_all = list(range(0, 480*640))
        # self.list_all_1 = self.list_all[:]
        # random.shuffle(self.list_all_1)
        
        self.val_psnr = []
        
        self.num = 0
        
    def _set_grid_resolution(self, num_voxels, mpi_depth, xyz_max, xyz_min):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        self.xyz_max = xyz_max
        self.xyz_min = xyz_min
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth     # W,H,D
        self.voxel_size_ratio = 256. / mpi_depth
        print('world_size:      ', self.world_size)
        print('voxel_size_ratio:', self.voxel_size_ratio)
        
    def _set_grid_resolution_blender(self, num_voxels, xyz_max, xyz_min):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.xyz_max = xyz_max
        self.xyz_min = xyz_min
        
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        
    def decode_batch(self, batch):

        # rays = batch['rays'] # (B, 9)
        # rgbs = batch['rgbs'] # (B, 3)
        # image_t = batch['image_t']
        # view = batch['view']
        # pixel_choose = batch['pixel_choose']
        # image_t = batch['time']
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

        return rays, rgbs, image_t, view
    def unpreprocess(self, data, shape=(1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    
    def forward(self, rays, world_size, grid_bounds):
        import time
        # torch.cuda.synchronize()
        # start = time.time()
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]  
        results = defaultdict(list)
        for i in range(0, B, self.args.chunk):
            rendered_ray_chunks = \
                render_sh_sample(self.models,
                            self.embeddings,
                            rays[i:i+self.args.chunk],   #[32768, 8]
                            world_size,
                            grid_bounds,
                            self.args.N_samples,
                            self.args.use_disp,
                            self.args.perturb,
                            self.args.noise_std,
                            self.args.N_importance,
                            self.args.chunk, # chunk size is effective in val mode  32768
                            self.train_dataset.white_back, 
                            
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

    def prepare_data(self):
        dataset = dataset_dict[self.args.dataset_name]
        # self.train_dataset = dataset(split='train', **kwargs)
        # self.val_dataset = dataset(split='val', **kwargs)
        train_dir = val_dir = self.args.root_dir
        # self.train_dataset = dataset(root_dir=train_dir, split='train', max_len=-1)
        # self.val_dataset   = dataset(root_dir=val_dir, split='val', max_len=10)
        self.train_dataset = dataset(root_dir=train_dir, split='train', img_wh = self.args.img_wh)
        self.xyz_min, self.xyz_max = self.train_dataset.get_box()
        self.coor_min = (self.box_min - torch.tensor((0,0,0)))/torch.tensor((self.size[0]-1,self.size[1]-1,self.size[2]-1))*(self.xyz_max - self.xyz_min) + self.xyz_min
        self.coor_max = (self.box_max - torch.tensor((0,0,0)))/torch.tensor((self.size[0]-1,self.size[1]-1,self.size[2]-1))*(self.xyz_max - self.xyz_min) + self.xyz_min
        self.grid_bounds = [self.coor_min.cuda(), self.coor_max.cuda()]        
        self._set_grid_resolution_blender(self.args.num_voxels, self.coor_max.cuda(), self.coor_min.cuda())
        # self.grid_bounds = [self.xyz_min.cuda(), self.xyz_max.cuda()]
        # self.world_size = torch.tensor([256, 256, 256])
        # self.output_feature_grid = torch.zeros((self.world_size.prod(), 28), device = device)
        # self.output_feature_grid = torch.nn.Parameter(torch.zeros(self.world_size.prod(), 28).cuda(),requires_grad = True)
        self.model_HashSiren = HashMlp(hash_mod = True,
                 hash_table_length = self.world_size.prod(),
                 in_features = self.args.in_features, 
                 hidden_features = self.args.hidden_features, 
                 hidden_layers = self.args.hidden_layers, 
                 out_features = self.args.out_features,
                 outermost_linear=True).cuda()
        # for i in self.model_HashSiren.net.parameters():
        #     i.requires_grad=False
        # self.model_HashSiren.net.requires_grad = False
        self.models = [self.model_HashSiren]
        self.val_dataset   = dataset(root_dir=val_dir, split='val', img_wh = self.args.img_wh)
    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.args, self.models)
        # self.optimizer = torch.optim.Adam(list(self.model_HashSiren.parameters()), lr=self.args.lr_NeRF, 
        #                  weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args, self.optimizer)

        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=0,
                        #   batch_size=self.args.batch_size,
                          batch_size=1024*2,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        import time
        # if self.trainer.current_epoch%300 == 0 and batch_nb== 0:
        #     random.shuffle(self.img_pixel)
        # iter = self.trainer.current_epoch%300
        
        # batch = batch[0]

        # log = {'lr1': get_learning_rate(self.optimizer1),
        #         'lr2': get_learning_rate(self.optimizer2)}
        # torch.cuda.synchronize()
        # start1 = time.time() 
        # self.back1 = time.time()
        # print("backward :",self.back1-self.back0)
        # self.start = time.time() 
        log = {'lr1': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        
        # flame = batch['flame']
        # if self.trainer.current_epoch == 0 and batch_nb== 0:
        #     self.grid_bounds = batch['grid_bounds']
        #     self.grid_bounds[0] = self.grid_bounds[0][0]
        #     self.grid_bounds[1] = self.grid_bounds[1][0]
        #     xyz_min = self.grid_bounds[0].squeeze()
        #     xyz_max = self.grid_bounds[1].squeeze()
        #     self._set_grid_resolution_blender(self.args.num_voxels, xyz_max, xyz_min)
           
           
           
            # self.model_HashSiren.HashTable(self.world_size.prod())
            
        # pixel_choose = self.img_pixel[(iter)*1024 : (iter+1)*1024]
        # rays = rays.squeeze()[pixel_choose]
        # rgbs = rgbs.squeeze()[pixel_choose]
        rays = rays.squeeze()
        rgbs = rgbs.squeeze()
        # torch.cuda.synchronize()
        # start2 = time.time()  
        # print("preparing:",start2-start1)
        # output_feature = self.model_HashSiren(rgbs)
        
        # out1 = output_feature[:,:54]
        # oyt = self.model_MLP_dir(out1)
        # ls = MSELoss1()
        # loss = ls(oyt, torch.ones_like(oyt))
        
        # torch.cuda.synchronize()
        # start = time.time()  
        # self.world_size = torch.tensor([256,256,256])
        results = self(rays, self.world_size, self.grid_bounds)
        # torch.cuda.synchronize()
        # end1 = time.time()  
        # print("forward  :",end1-start)
        # tv_sigma = results['tv_sigma']
        # tv_feature_ = results['tv_feature_']
        
        
        # print(self.world_size)   
        
        
        
        # ls = MSELoss1()
        # loss = ls(results['rgb_coarse'], torch.ones_like(results['rgb_coarse']))
        # loss = ls(results, torch.ones_like(results))
        
        log['train/loss'] = loss = self.loss(results, rgbs)
        # log['train/loss'] = loss = self.loss(results, rgbs) + self.tvloss_sigma(tv_sigma) + self.tvloss_sh(tv_feature_)
        if torch.any(torch.isnan(loss)):
            pdb.set_trace() 
            results = self(rays, self.world_size, self.grid_bounds)
            print("loss:", self.loss(results, rgbs))


        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        self.log('train/loss', loss.item(), prog_bar=True)
        self.log('train/psnr', psnr_.item(), prog_bar=True)

        # torch.cuda.synchronize()
        # end = time.time()  
        # self.back0 = time.time()  
        # print("total    :",end-start1)
        return {'loss': loss}
        # return {'loss': loss,
        #         'progress_bar': {'train/psnr': psnr_},
        #         'log': log
        #        }

    def validation_step(self, batch, batch_nb):
        
        # self.idx += 1
        # if self.idx == 1:
        #     self.idx_gpus = 0
        # else:
        #     self.idx_gpus = (self.idx-2)//4 + 1
        # # self.idx_gpus = self.idx//8

        self.idx += 1
        if self.idx == 1:
            self.idx_gpus = 0
        else:
            self.idx_gpus = (self.idx - 2)//8 + 1
        if self.trainer.current_epoch == 0 and batch_nb== 0:
            self._set_grid_resolution_blender(self.args.num_voxels, self.xyz_max, self.xyz_min)
        self.idx_gpus += 1
        
        rays, rgbs = self.decode_batch(batch)
        
        
        rays = rays.squeeze() # (H*W, 3) [160000, 8]
        rgbs = rgbs.squeeze() # (H*W, 3)
        # if self.trainer.current_epoch == 0:
        #     self.grid_bounds = batch['grid_bounds']
        #     xyz_min = self.grid_bounds[0].squeeze()
        #     xyz_max = self.grid_bounds[1].squeeze()
        #     self._set_grid_resolution(self.args.num_voxels, self.args.mpi_depth, xyz_max, xyz_min)
        # self.world_size = torch.tensor([256,256,256])
        results = self(rays, self.world_size, self.grid_bounds)      
          
        # results = self(rays.float())
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        # log['psnr_nerv'] = psnr(rgbs_NeRV, rgbs)

        # img = results['rgb_coarse'].reshape(400,400,3).permute(2,0,1).cpu()
        img = results[f'rgb_{typ}'].reshape(*self.image_shape.shape).permute(2,0,1).cpu()
        # img1 = rgbs.reshape(400,400,3).permute(2,0,1).cpu()
        img1 = rgbs.reshape(*self.image_shape.shape).permute(2,0,1).cpu()

        img1 = img1.unsqueeze(0)
        img = img.unsqueeze(0)
        # img2 = img2.unsqueeze(0)
        img_vis = torch.cat((img1,img),dim=0).permute(2,0,3,1).reshape(img1.shape[2],-1,3).numpy()
        os.makedirs(f'{self.save_path}/hash_table/val_img/{self.args.exp_name}/',exist_ok=True)
        imageio.imwrite(f'{self.save_path}/hash_table/val_img/{self.args.exp_name}/{self.idx_gpus:02d}_{batch_nb:02d}.png', (img_vis*255).astype('uint8'))
        

        
        return log

    def validation_epoch_end(self, outputs):
        
        # self.point_list = self.list_all_1[(self.flag_epoch1 - 1) * 1024 : self.flag_epoch1 * 1024]
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        # psnr_nerv = torch.stack([x['psnr_nerv'] for x in outputs]).mean()

        self.log('val_loss', mean_loss.item(), prog_bar=True)
        self.log('val_psnr', mean_psnr.item(), prog_bar=True)
        # self.log('psnr_nerv', psnr_nerv.item(), prog_bar=True)
        if self.flag_epoch1 < 4:
            self.val_psnr += [mean_psnr]
            self.save_ckpt(mean_psnr, self.flag_epoch1)
        else:
            min_psnr = min(self.val_psnr)
            if mean_psnr > min_psnr :
                idx = self.val_psnr.index(min_psnr)
                self.val_psnr[idx] = mean_psnr
                self.save_ckpt(mean_psnr, idx)
        self.flag_epoch1 += 1 
        if self.trainer.current_epoch == 1:
            self.save_ckpt(mean_psnr, "one epoch")
        self.PSNR = mean_psnr
        return 

    def save_ckpt(self, psnr, name='final'):
        
        save_dir = f'{self.save_path}/hash_table/checkpoints/{self.args.exp_name}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/HashTable_{name}.tar'
        ckpt = {
                'val_PSNR' : psnr,
                'model_HashSiren' :self.models[0].state_dict(),}
                # 'model_MLP_dir' :self.models[1].state_dict()}
       
        torch.save(ckpt, path)
        print('Saved checkpoints at', path)

# 写入json文件
def write_json_data(path, params):
    # 使用写模式，名称定义为r
    #其中路径如果和读json方法中的名称不一致，会重新创建一个名称为该方法中写的文件名
    json_str=json.dumps(params,indent=4,ensure_ascii=False)
    with open(path+'record.json', 'w', encoding='utf-8') as json_file:
        # 将dict写入名称为r的文件中
        json_file.write(json_str)

if __name__ == '__main__':
    with torch.cuda.device(0):
        args = config_parser()
        system = NeRFSystem(args)
        a = os.path.join(f'{system.save_path}/hash_table/checkpoints/{args.exp_name}/ckpts/','{epoch:02d}')
        dirpath = f'{system.save_path}/hash_table/checkpoints/{args.exp_name}/ckpts/'
        # filename = '{epoch:02d}'
        filename = '{epoch:02d}-{val_loss:.3f}'
        import json
        os.makedirs(dirpath,exist_ok=True)
        record_code = {
            'name' : args.exp_name,
            'description' : "网格160,28维hash,2048个采样点,再次采样,把box变小,出了box直接设置为1,MLP为2*64,送入MLP网络为当前所需网格点"
        }
        write_json_data(dirpath, record_code)
        # early_stop_callback = (
        #     EarlyStopping(
        #         monitor = 'val/loss_mean',
        #         patience = 15,
        #         mode = 'min')
        #     )
        # checkpoint_callback = ModelCheckpoint(dirpath = dirpath,
        #                                       filename = filename,
        #                                       monitor='val_psnr',
        #                                       mode='max',
        #                                       save_top_k=5,)
                                            #   auto_insert_metric_name=False)

        logger = TestTubeLogger(
            save_dir=f'{system.save_path}/hash_table/logs',
            name=args.exp_name,
            debug=False,
            create_git_tag=False
        )
        
        profiler = AdvancedProfiler(dirpath=f'{system.save_path}/hash_table/logs/{args.exp_name}', filename="perf_logs")
        trainer = Trainer(max_epochs=args.num_epochs,
                        #   automatic_optimization = False,
                        #   checkpoint_callback=checkpoint_callback,
                        #   callbacks=[checkpoint_callback, early_stop_callback],
                        #   callbacks=[checkpoint_callback],
                        #   resume_from_checkpoint=args.ckpt_path,
                        logger=logger,
                        #   early_stop_callback=None,
                        weights_summary=None,
                        progress_bar_refresh_rate=1,
                        #   gpus=args.num_gpus,
                        gpus=[0],
                        distributed_backend='ddp' if args.num_gpus>1 else None,
                        num_sanity_val_steps = 0,     #训练之前进行校验
                        check_val_every_n_epoch = 1,   #一个epoch校验一次
                        # val_check_interval=0.1,      #0.1个epoch校验一次
                        precision=16, 
                        benchmark=True,
                        log_every_n_steps = 50,    #每隔1次迭代记录一下logger
                        profiler=profiler)

        # pdb.set_trace()
        trainer.fit(system)
        system.save_ckpt(psnr = system.PSNR)
        torch.cuda.empty_cache()

###在动态数据集上加东西   图片一般是(h, w, 3)
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
from models.nerf import Embedding, NeRF, LatentCode
from models.rendering import render_rays, render_rays1, render_rays2
from models.model_nerv import *

# optimizer, scheduler, visualization, NeRV utils
from utils import *
from utils import get_optimizer1
from utils.NeRV import *
import torch.optim as optim

# losses
from losses import loss_dict
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
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2 ,3"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class NeRFSystem(LightningModule):    
    def __init__(self, args):
        super(NeRFSystem, self).__init__()
        self.automatic_optimization=False       #把自动优化关掉
        self.args = args
        self.idx = 0
        self.loss = loss_dict[args.loss_type]()
        self.loss1 = loss_dict['mse1']()
        self.NeRV_ckpt = args.weight + args.view_choose + args.NeRV_model
        self.N_NeRV = 10
        self.N_NeRF = 100
        self.NeRV_ckpt = []
        for view in range(18):
            view_choose = f'/cam{view:02d}'
            self.NeRV_ckpt += [args.weight + view_choose + args.NeRV_model]
        
        self.NeRV_list = []
        # self.embedding_xyz = Embedding(3, 10) # 10 is the default number
            #NeRV对t进行pe编码
        self.PE = PositionalEncoding(args.embed)  
        args.embed_length = self.PE.embed_length
        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]
        self.list_log = []
        self.img_wh = self.args.img_wh
        self.image_shape = torch.zeros(self.img_wh[1], self.img_wh[0], 3)
        args.single_res = True
        self.epoch_my = 0
        self.iter_num = 0


        self.img_pixel = list(range(0, 480 * 640))
        # random.shuffle(self.img_pixel)


        self.Ir_tensor = torch.zeros(18)
        # self.model_NeRV_cam00 = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        # num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        # stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid)
        
        self.NeRVs = [Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid).cuda() for i in range(18)]
        # self.NeRVs = nn.ModuleList(NeRVs)
        # for name, p in self.NeRVs.named_parameters():
        #     print(name)
        
        # for name, module in self.NeRVs.named_modules():
        #     if name == '0':
        #         print(module)
        # self.model_NeRV = {}
        # for x in range(18):
        #     self.model_NeRV[f'self.model_NeRV_cam{x:02d}'] = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        # num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        # stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid).cuda()
        
        # self.NeRV_keys = []
        # for i, key in enumerate(self.model_NeRV.keys()):
        #     self.NeRV_keys += [key]
        #     # model = self.model_NeRV[key]
        #     # exec('model_{} = {}'.format(i, self.model_NeRV_cam00))
        #     # setattr(self, f"model_{i:02d}", self.model_NeRV_cam00)
        #     # model_0 = self.model_NeRV_cam00
        #     if self.NeRV_ckpt[i] != 'None':
        #         # model = self.model_NeRV[f'self.model_NeRV_cam{i:02d}']
                
        #         print("=> loading checkpoint '{}'".format(self.NeRV_ckpt[i]))
        #         checkpoint_path = self.NeRV_ckpt[i]
        #         checkpoint = torch.load(checkpoint_path, map_location='cpu')
        #         orig_ckt = checkpoint['state_dict']
        #         new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        #         if 'module' in list(orig_ckt.keys())[0] and not hasattr(self.model_NeRV[key], 'module'):
        #             new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
        #             self.model_NeRV[key].load_state_dict(new_ckt)
        #         elif 'module' not in list(orig_ckt.keys())[0] and hasattr(self.model_NeRV[key], 'module'):
        #             self.model_NeRV[key].module.load_state_dict(new_ckt)
        #         else:
        #             self.model_NeRV[key].load_state_dict(new_ckt)
        #         print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))
        


        for i in range(len(self.NeRVs)):
            if self.NeRV_ckpt[i] != 'None':                
                print("=> loading checkpoint '{}'".format(self.NeRV_ckpt[i]))
                checkpoint_path = self.NeRV_ckpt[i]
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                orig_ckt = checkpoint['state_dict']
                opt = checkpoint['optimizer']
                Ir = opt['param_groups'][0]['lr']
                self.Ir_tensor[i] = Ir
                new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
                if 'module' in list(orig_ckt.keys())[0] and not hasattr(self.NeRVs[i], 'module'):
                    new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
                    self.NeRVs[i].load_state_dict(new_ckt)
                elif 'module' not in list(orig_ckt.keys())[0] and hasattr(self.NeRVs[i], 'module'):
                    self.NeRVs[i].module.load_state_dict(new_ckt)
                else:
                    self.NeRVs[i].load_state_dict(new_ckt)
                print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))
        args.lr_NeRV = self.Ir_tensor.mean()
        # if args.ckpt_path is not None:
        #     ckpt_nerf = torch.load(args.ckpt_path)
        self.nerf_coarse = NeRF()
        load_ckpt(self.nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    
        # self.nerf_coarse.load_state_dict(ckpt_nerf['nerf_coarse_state_dict'])
        self.models = [self.nerf_coarse]
        if args.N_importance > 0:
            self.nerf_fine = NeRF()
            load_ckpt(self.nerf_fine, args.ckpt_path, model_name='nerf_fine')
            # self.nerf_fine.load_state_dict(ckpt_nerf['nerf_fine_state_dict'])
            self.models += [self.nerf_fine]
        self.LatentCode = LatentCode()
        load_ckpt(self.LatentCode, args.ckpt_path, model_name='LatentCode')
        # self.LatentCode.load_state_dict(ckpt_nerf['latent_code'])
        self.models += [self.LatentCode]
        self.t_normalize = 0
        self.flag_image_num = 0
        self.flag_epoch1 = 0
        self.flag_epoch2 = 0

        self.list_all = list(range(0, 480*640))
        self.list_all_1 = self.list_all[:]
        random.shuffle(self.list_all_1)
        
        self.val_psnr = []
        
        self.num = 0
    def decode_batch(self, batch):

        rays = batch['rays'] # (B, 9)
        rgbs = batch['rgbs'] # (B, 3)
        image_t = batch['image_t']
        flag_Interpolation = batch['flag']
        view = batch['view']
        # pixel_choose = batch['pixel_choose']
        # image_t = batch['time']

        return rays, rgbs, image_t, flag_Interpolation, view
    def unpreprocess(self, data, shape=(1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std
    def forward(self, rays, t_normalize = 0):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]  #160000
        results = defaultdict(list)
        for i in range(0, B, self.args.chunk):
            rendered_ray_chunks = \
                render_rays2(self.models,
                            self.embeddings,
                            rays[i:i+self.args.chunk],   #[32768, 8]
                            self.args.N_samples,
                            self.args.use_disp,
                            self.args.perturb,
                            self.args.noise_std,
                            self.args.N_importance,
                            self.args.chunk, # chunk size is effective in val mode  32768
                            self.train_dataset.white_back, 
                            t_normalize = t_normalize,
                            test_time=False
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]   #k  'rgb_coarse'  v为数值

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.args.dataset_name]
        # self.train_dataset = dataset(split='train', **kwargs)
        # self.val_dataset = dataset(split='val', **kwargs)
        train_dir = val_dir = self.args.root_dir
        self.train_dataset = dataset(root_dir=train_dir, split='train', max_len=-1)
        self.val_dataset   = dataset(root_dir=val_dir, split='val', max_len=10)

    def configure_optimizers(self):
        self.optimizer1 = get_optimizer(self.args, self.models)
        scheduler1 = get_scheduler(self.args, self.optimizer1)
        self.optimizer_NeRV = []
        self.scheduler_NeRV = []
        # self.optimizer2 = get_optimizer1(self.args, [self.NeRVs[0]])
        for i in range(18):
            optimizer_NeRV = get_optimizer1(self.args, self.Ir_tensor, [self.NeRVs[i]], i)
            self.optimizer_NeRV += [optimizer_NeRV]
            self.scheduler_NeRV += [CosineAnnealingLR(optimizer_NeRV, T_max=self.args.num_epochs, eta_min=1e-8)]
        # self.optimizer3 = get_optimizer1(self.args, [self.NeRVs[1]])
        # scheduler2 = CosineAnnealingLR(self.optimizer2, T_max=self.args.num_epochs, eta_min=1e-8)
        # scheduler3 = CosineAnnealingLR(self.optimizer3, T_max=self.args.num_epochs, eta_min=1e-8)
        # self.optimizer2 = optim.Adam(model.parameters(), betas=(args.beta, 0.999))
        return [self.optimizer1]+self.optimizer_NeRV, [scheduler1]+self.scheduler_NeRV
        return [self.optimizer1, self.optimizer2, self.optimizer3], [scheduler1, scheduler2, scheduler3]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=0,
                        #   batch_size=self.args.batch_size,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb, optimizer_idx):
        # self.iter_num  = batch_nb
        # if self.iter_num == 18*90-1:
        #     self.iter_num = 0
        #     self.epoch_my += 1
        #     if self.epoch_my == 299:
        #         self.epoch_my = 0
        #         random.shuffle(self.img_pixel)
        if self.trainer.current_epoch%300 == 0 and batch_nb== 0:
            random.shuffle(self.img_pixel)
        iter = self.trainer.current_epoch%300
        


        # log = {'lr1': get_learning_rate(self.optimizer1),
        #         'lr2': get_learning_rate(self.optimizer2)}
        log = {'lr1': get_learning_rate(self.optimizer1)}
        rays, rgbs, image_t, flag_Interpolation, view = self.decode_batch(batch)
        pixel_choose = self.img_pixel[(iter)*1024 : (iter+1)*1024]
        rays = rays.squeeze()[pixel_choose]
        # view = 0
        # rgbs = rgbs[pixel_choose]
        if flag_Interpolation == True:    #gt要从NeRV网络中得到
            # with torch.no_grad():
            # self.models = self.models[:3]
            loss_weight = 0.01
            # loss_weight1 = 0
            model_NeRV = self.NeRVs[view]
            embed_t = self.PE(image_t)
            embed_t = embed_t.cuda(non_blocking=True)
            rgbs = model_NeRV(embed_t.float())[0].squeeze().reshape(3, -1).permute(1, 0)
            rgbs = rgbs[pixel_choose]
            # rgbs_NeRV = torch.tensor(0, device=rgbs.device)
            # self.models += [model_NeRV]
        if flag_Interpolation == False:
            rgbs = rgbs.squeeze()
            rgbs = rgbs[pixel_choose]
            loss_weight = 0.8
            loss_weight1 = 0.2
            model_NeRV = self.NeRVs[view]
            embed_t = self.PE(image_t)
            embed_t = embed_t.cuda(non_blocking=True)
            rgbs_NeRV = model_NeRV(embed_t.float())[0].squeeze().reshape(3, -1).permute(1, 0)
            rgbs_NeRV = rgbs_NeRV[pixel_choose]



        t_normalize = image_t
        rays = rays.squeeze()[:, :8]
        
        results = self(rays.float(), t_normalize = t_normalize)  #[1024,3]
        if flag_Interpolation == False:
            # if torch.any(torch.isnan(results)):
            #     results = results[~torch.isnan(results)]
            #     rgbs = rgbs[~torch.isnan(results)]
            #     rgbs_NeRV = rgbs_NeRV[~torch.isnan(results)]
            if torch.any(torch.isnan(rgbs_NeRV)):
                pdb.set_trace() 
                results['rgb_coarse'] = results['rgb_coarse'][~torch.isnan(rgbs_NeRV)]
                results['rgb_fine'] = results['rgb_fine'][~torch.isnan(rgbs_NeRV)]
                rgbs = rgbs[~torch.isnan(rgbs_NeRV)]
                rgbs_NeRV = rgbs_NeRV[~torch.isnan(rgbs_NeRV)]
            log['train/loss'] = loss = self.loss(results, rgbs) * loss_weight + self.loss1(rgbs_NeRV, rgbs) * loss_weight1
        else:
            if torch.any(torch.isnan(rgbs)):
                pdb.set_trace() 
                results['rgb_coarse'] = results['rgb_coarse'][~torch.isnan(rgbs)]
                results['rgb_fine'] = results['rgb_fine'][~torch.isnan(rgbs)]
                rgbs = rgbs[~torch.isnan(rgbs)]
            log['train/loss'] = loss = self.loss(results, rgbs) * loss_weight
        if torch.any(torch.isnan(loss)):
            pdb.set_trace() 

            print("flag_Interpolation:", flag_Interpolation)
            print("loss:", self.loss(results, rgbs))


        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        self.log('train/loss', loss.item(), prog_bar=True)
        self.log('train/psnr', psnr_.item(), prog_bar=True)
        # self.logger.experiment.add_images
        # for p in self.NeRVs[1].parameters():
        #     p.requires_grad = False
        # opt_NeRF, opt_NeRV_1, opt_NeRV_2 = self.optimizers(use_pl_optimizer=True)
        opt = self.optimizers(use_pl_optimizer=True)
        
        opt_NeRF = opt[0]
        
        opt_NeRV = opt[1:]
        
        opt_NeRF.zero_grad()
        opt_NeRV[view].zero_grad()

        self.manual_backward(loss)
        opt_NeRF.step()
        opt_NeRV[view].step()
        
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % self.N_NeRF == 0:
            sch = self.lr_schedulers()
            sch_NeRF = sch[0]
        # if self.trainer.is_last_batch :
            sch_NeRF.step()
            # sch_NeRV_1.step()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % self.N_NeRV == 0:
            sch = self.lr_schedulers()
            sch_NeRV = sch[1:]
            sch_NeRV[view].step()
        return {'loss': loss}
        return {'loss': loss,
                'progress_bar': {'train/psnr': psnr_},
                'log': log
               }

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


        rays, rgbs, image_t, flag_Interpolation, view = self.decode_batch(batch)
        # model_NeRV = self.NeRVs[view]
        # embed_t = self.PE(image_t)
        # embed_t = embed_t.cuda(non_blocking=True)
        # rgbs_NeRV = model_NeRV(embed_t.float())[0].squeeze().reshape(3, -1).permute(1, 0)
        
        
        time = int(image_t*30)
        # # print("val_t:",image_t)
        # rays = rays.squeeze() # (H*W, 3) [160000, 8]
        # rgbs = rgbs.squeeze() # (H*W, 3)
        # t_num1 = torch.tensor(30.0)
        # image_t = image_t*torch.ones_like(rays[:, :1])
        # t_normalize = 2*image_t.type(torch.FloatTensor)/(t_num1.type(torch.FloatTensor))-1
        
        rays = rays.squeeze() # (H*W, 3) [160000, 8]
        rgbs = rgbs.squeeze() # (H*W, 3)
        t_normalize = image_t

        results = self(rays.float(), t_normalize = t_normalize)  #[160000,3]
        
        # results = self(rays.float())
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        # log['psnr_nerv'] = psnr(rgbs_NeRV, rgbs)

        # img = results['rgb_coarse'].reshape(400,400,3).permute(2,0,1).cpu()
        img = results[f'rgb_{typ}'].reshape(*self.image_shape.shape).permute(2,0,1).cpu()
        # img1 = rgbs.reshape(400,400,3).permute(2,0,1).cpu()
        img1 = rgbs.reshape(*self.image_shape.shape).permute(2,0,1).cpu()
        # img2 = rgbs_NeRV.reshape(*self.image_shape.shape).permute(2,0,1).cpu()
        # img = img.cuda()
        # img1 = img1.cuda()
        # img = self.unpreprocess(img).squeeze().cpu()
        # img1 = self.unpreprocess(img1).squeeze().cpu()
        
        # toPIL = transforms.ToPILImage()
        # pic = toPIL(img)
        # # pic.save(f'pre_{batch_nb}.jpg')
        # pic = toPIL(img1)
        # pic.save(f'gt_{batch_nb}.jpg')
        img1 = img1.unsqueeze(0)
        img = img.unsqueeze(0)
        # img2 = img2.unsqueeze(0)
        img_vis = torch.cat((img1,img),dim=0).permute(2,0,3,1).reshape(img1.shape[2],-1,3).numpy()
        os.makedirs(f'/data1/liufengyi/get_results/non_synchronized_NeRF2/runs_new_try1/{self.args.exp_name}/{self.args.exp_name}/',exist_ok=True)
        imageio.imwrite(f'/data1/liufengyi/get_results/non_synchronized_NeRF2/runs_new_try1/{self.args.exp_name}/{self.args.exp_name}/{time:02d}_{self.idx_gpus:02d}.png', (img_vis*255).astype('uint8'))
        
        # if batch_nb == 0:
        #     W, H = self.args.img_wh
        #     img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
        #     img = img.permute(2, 0, 1) # (3, H, W)
        #     img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        #     depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        #     stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
        #     self.logger.experiment.add_images('val/GT_pred_depth',
        # 
        #                                        stack, self.global_step)

        
        return log

    def validation_epoch_end(self, outputs):
        self.flag_epoch1 += 1 
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
        
        return 
        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }

    def save_ckpt(self, psnr, name='final'):
        save_dir = f'/data1/liufengyi/get_results/non_synchronized_NeRF2/runs_new/{self.args.exp_name}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path1 = f'{save_dir}/NeRV_{name}.tar'
        path2 = f'{save_dir}/NeRF_{name}.tar'
        ckpt1 = {}
        ckpt1['val_PSNR'] = psnr
        for i in range(18):
            ckpt1[f'NeRVs_{i}'] = self.NeRVs[i].state_dict()
        
        # ckpt1 = {
        #     # 'global_step': self.global_step,
        #     # 'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
        #     # 'network_mvs_state_dict': self.MVSNet.state_dict()}
        #         'val_PSNR' : psnr,
        #         'NeRVs_0' :self.NeRVs[0].state_dict(),
        #         'nerf_fine_state_dict' :self.models[1].state_dict(),
        #         'latent_code': self.models[2].state_dict()}
        ckpt2 = {
                'val_PSNR' : psnr,
                'nerf_coarse_state_dict' :self.models[0].state_dict(),
                'nerf_fine_state_dict' :self.models[1].state_dict(),
                'latent_code': self.models[2].state_dict()}
        torch.save(ckpt1, path1)
        torch.save(ckpt2, path2)
        print('Saved checkpoints1 at', path1)
        print('Saved checkpoints2 at', path2)

if __name__ == '__main__':
    with torch.cuda.device(2):
        args = config_parser()
        system = NeRFSystem(args)
        a = os.path.join(f'/data1/liufengyi/get_results/non_synchronized_NeRF2/runs_new/{args.exp_name}/ckpts/','{epoch:02d}')
        dirpath = f'/data1/liufengyi/get_results/non_synchronized_NeRF2/runs_new/{args.exp_name}/ckpts/'
        # filename = '{epoch:02d}'
        filename = '{epoch:02d}-{val_loss:.3f}'
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
            save_dir="/data1/liufengyi/get_results/non_synchronized_NeRF2/logs",
            name=args.exp_name,
            debug=False,
            create_git_tag=False
        )

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
                        gpus=[2],
                        distributed_backend='ddp' if args.num_gpus>1 else None,
                        num_sanity_val_steps = 1,     #训练之前进行校验
                        check_val_every_n_epoch = 100,   #一个epoch校验一次
                        # val_check_interval=0.25,      #0.1个epoch校验一次
                        precision=16, 
                        benchmark=True,
                        log_every_n_steps = 50,)    #每隔1次迭代记录一下logger
                        #   profiler=args.num_gpus==1)

        # pdb.set_trace()
        trainer.fit(system)
        system.save_ckpt()
        torch.cuda.empty_cache()

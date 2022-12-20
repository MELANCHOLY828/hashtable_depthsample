###在动态数据集上加东西   图片一般是(h, w, 3)
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
from utils.NeRV import *

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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2 ,3"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# device = torch.device("cuda")
class NeRFSystem(LightningModule):    
    def __init__(self, args):
        super(NeRFSystem, self).__init__()
        self.args = args
        self.idx = 0
        self.loss = loss_dict[args.loss_type]()
        self.NeRV_ckpt = args.weight + args.view_choose + args.NeRV_model
        
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
        # self.model_NeRV_cam00 = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        # num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        # stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid)
        
        
        # NeRVs = [Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        # num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        # stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid) for i in range(18)]
        # self.NeRVs = nn.ModuleList(NeRVs)


        # for name, p in self.NeRVs.named_parameters():
        #     print(name)
        
        # for name, module in self.NeRVs.named_modules():
        #     if name == '0':
        #         print(module)
        self.model_NeRV = {}
        for x in range(18):
            self.model_NeRV[f'self.model_NeRV_cam{x:02d}'] = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid).cuda()
        # for i in range(1, 18):
            # f'self.model_NeRV_cam_{i:02d}' = 
        self.NeRV_keys = []
        # for i in range(len(self.NeRV_ckpt)):
        for i, key in enumerate(self.model_NeRV.keys()):
            self.NeRV_keys += [key]
            # model = self.model_NeRV[key]
            # exec('model_{} = {}'.format(i, self.model_NeRV_cam00))
            # setattr(self, f"model_{i:02d}", self.model_NeRV_cam00)
            # model_0 = self.model_NeRV_cam00
            if self.NeRV_ckpt[i] != 'None':
                # model = self.model_NeRV[f'self.model_NeRV_cam{i:02d}']
                
                print("=> loading checkpoint '{}'".format(self.NeRV_ckpt[i]))
                checkpoint_path = self.NeRV_ckpt[i]
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                orig_ckt = checkpoint['state_dict']
                new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
                self.model_NeRV[key].to(device)
                if 'module' in list(orig_ckt.keys())[0] and not hasattr(self.model_NeRV[key], 'module'):
                    new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
                    self.model_NeRV[key].load_state_dict(new_ckt)
                elif 'module' not in list(orig_ckt.keys())[0] and hasattr(self.model_NeRV[key], 'module'):
                    self.model_NeRV[key].module.load_state_dict(new_ckt)
                else:
                    self.model_NeRV[key].load_state_dict(new_ckt)
                print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))
        self.nerf_coarse = NeRF()
        load_ckpt(self.nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
        
        self.models = [self.nerf_coarse]
        if args.N_importance > 0:
            self.nerf_fine = NeRF()
            load_ckpt(self.nerf_fine, args.ckpt_path, model_name='nerf_fine')
            self.models += [self.nerf_fine]
        self.LatentCode = LatentCode()
        load_ckpt(self.LatentCode, args.ckpt_path, model_name='LatentCode')
        self.models += [self.LatentCode]
        self.t_normalize = 0
        self.flag_image_num = 0
        self.flag_epoch1 = 0
        self.flag_epoch2 = 0

        self.list_all = list(range(0, 480*640))
        self.list_all_1 = self.list_all[:]
        random.shuffle(self.list_all_1)
        

        
        self.num = 0
    def decode_batch(self, batch):

        rays = batch['rays'] # (B, 9)
        rgbs = batch['rgbs'] # (B, 3)
        image_t = batch['image_t']
        flag_Interpolation = batch['flag']
        view = batch['view']
        pixel_choose = batch['pixel_choose']
        # image_t = batch['time']

        return rays, rgbs, image_t, flag_Interpolation, view, pixel_choose
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
        self.optimizer = get_optimizer(self.args, self.models)
        scheduler = get_scheduler(self.args, self.optimizer)
        
        return [self.optimizer], [scheduler]

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
    
    def training_step(self, batch, batch_nb):
        
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs, image_t, flag_Interpolation, view, pixel_choose = self.decode_batch(batch)
        if flag_Interpolation == True:    #gt要从NeRV网络中得到
            with torch.no_grad():
                loss_weight = 0.01
                model_NeRV = self.model_NeRV[self.NeRV_keys[view]]
                embed_t = self.PE(image_t)
                embed_t = embed_t.cuda(non_blocking=True)
                rgbs = model_NeRV(embed_t.float())[0].squeeze().reshape(3, -1).permute(1, 0)
                rgbs = rgbs[pixel_choose]
        if flag_Interpolation == False:
            # rgbs = rgbs.squeeze()
            model_NeRV = self.model_NeRV[self.NeRV_keys[view]]
            embed_t = self.PE(image_t)
            embed_t = embed_t.cuda(non_blocking=True)
            rgbs = model_NeRV(embed_t.float())[0].squeeze().reshape(3, -1).permute(1, 0)
            rgbs = rgbs[pixel_choose]
            loss_weight = 1
        # image_t = rays[:, -1].unsqueeze(1)
        # t_num1 = torch.tensor(30.0)
        # t_normalize = 2*scene_t/(t_num1-1)-1  #0
        # t_normalize = 2*image_t.type(torch.FloatTensor)/(t_num1.type(torch.FloatTensor))-1
        # t_normalize = 0
        # list = random.sample(range(0,2704 * 2028),1024)
        t_normalize = image_t
        rays = rays.squeeze()[:, :8]
        
        # rgbs = rgbs[self.point_list]
        # rays = rays[self.point_list]
        
        results = self(rays.float(), t_normalize = t_normalize)  #[1024,3]
        log['train/loss'] = loss = self.loss(results, rgbs) * loss_weight
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        # w, h = self.hparams.img_wh
        # img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        # img_pred_ = (img_pred*255).astype(np.uint8)
        # dir_name = '/home/liufengyi/test/nerf_pl-master/nerf_pl-master'
        # i = 0
        # imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        self.log('train/loss', loss.item(), prog_bar=True)
        self.log('train/psnr', psnr_.item(), prog_bar=True)
        # self.logger.experiment.add_images

        return {'loss': loss}
        return {'loss': loss,
                'progress_bar': {'train/psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):

        self.idx += 1
        if self.idx == 1:
            self.idx_gpus = 0
        else:
            self.idx_gpus = (self.idx - 2)//8 + 1
        # else:
        #     self.idx_gpus = (self.idx-2)//4 + 1
        
        # self.idx_gpus = self.idx//8

        rays, rgbs, image_t, flag_Interpolation, view, pixel_choose = self.decode_batch(batch)
        
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
        # img = results['rgb_coarse'].reshape(400,400,3).permute(2,0,1).cpu()
        img = results[f'rgb_{typ}'].reshape(*self.image_shape.shape).permute(2,0,1).cpu()
        # img1 = rgbs.reshape(400,400,3).permute(2,0,1).cpu()
        img1 = rgbs.reshape(*self.image_shape.shape).permute(2,0,1).cpu()
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
        self.point_list = self.list_all_1[(self.flag_epoch1 - 1) * 1024 : self.flag_epoch1 * 1024]
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        self.log('val_loss', mean_loss.item(), prog_bar=True)
        self.log('val_psnr', mean_psnr.item(), prog_bar=True)

        return 
        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }

    def save_ckpt(self, name='final'):
        save_dir = f'/data1/liufengyi/get_results/non_synchronized_NeRF2/runs_new/{self.args.exp_name}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/{name}.tar'
        ckpt = {
            # 'global_step': self.global_step,
            # 'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
            # 'network_mvs_state_dict': self.MVSNet.state_dict()}
                'nerf_coarse_state_dict' :self.models[0].state_dict(),
                'nerf_fine_state_dict' :self.models[1].state_dict(),
                'latent_code': self.models[2].state_dict()}
        # if self.render_kwargs_train['network_fine'] is not None:
        #     ckpt['network_fine_state_dict'] = self.render_kwargs_train['network_fine'].state_dict()
        torch.save(ckpt, path)
        print('Saved checkpoints at', path)

if __name__ == '__main__':
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
    checkpoint_callback = ModelCheckpoint(dirpath = dirpath,
                                          filename = filename,
                                          monitor='val_psnr',
                                          mode='max',
                                          save_top_k=5,)
                                        #   auto_insert_metric_name=False)

    logger = TestTubeLogger(
        save_dir="/data1/liufengyi/get_results/non_synchronized_NeRF2/logs",
        name=args.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=args.num_epochs,
                    #   checkpoint_callback=checkpoint_callback,
                    #   callbacks=[checkpoint_callback, early_stop_callback],
                      callbacks=[checkpoint_callback],
                    #   resume_from_checkpoint=args.ckpt_path,
                      logger=logger,
                    #   early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                    #   gpus=args.num_gpus,
                      gpus=[2],
                      distributed_backend='ddp' if args.num_gpus>1 else None,
                      num_sanity_val_steps = 1,     #训练之前进行校验
                    #   check_val_every_n_epoch = 1,   #一个epoch校验一次
                      val_check_interval=0.25,      #0.1个epoch校验一次
                      precision=16, 
                      benchmark=True,
                      log_every_n_steps = 50,   )    #每隔1次迭代记录一下logger
                    #   profiler=args.num_gpus==1)

    # pdb.set_trace()
    trainer.fit(system)
    system.save_ckpt()
    torch.cuda.empty_cache()

###在动态数据集上加东西
import os, sys
from opt import config_parser
import torch
from collections import defaultdict
import random
from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays, render_rays1

# optimizer, scheduler, visualization
from utils import *

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeRFSystem(LightningModule):
    def __init__(self, args):
        super(NeRFSystem, self).__init__()
        self.args = args
        self.idx = 0
        self.loss = loss_dict[args.loss_type]()

        # self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_xyz = Embedding(4, 10)
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]
        self.list_log = []

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if args.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]
        self.t_normalize = 0
        self.list = random.sample(range(0,230400),230400)

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        scene_t = batch['scene_t']
        t_num1 = batch['t_num1']
        return rays, rgbs, scene_t, t_num1
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
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.args.chunk],   #[32768, 8]
                            self.args.N_samples,
                            self.args.use_disp,
                            self.args.perturb,
                            self.args.noise_std,
                            self.args.N_importance,
                            self.args.chunk, # chunk size is effective in val mode  32768
                            self.train_dataset.white_back, 
                            # t_normalize = t_normalize,
                            test_time=False
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]   #k  'rgb_coarse'  v为数值

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.args.dataset_name]
        kwargs = {'root_dir': self.args.root_dir,
                  'img_wh': tuple(self.args.img_wh)}
        if self.args.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.args.spheric_poses
            kwargs['val_num'] = self.args.num_gpus
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
                          num_workers=4,
                        #   batch_size=self.args.batch_size,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=True,
                          num_workers=1,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs, scene_t, t_num1 = self.decode_batch(batch)
        view = batch['view']
        
        # t_normalize = 2*scene_t/(t_num1-1)-1  #0
        t_normalize = 2*scene_t.type(torch.FloatTensor)/(t_num1.type(torch.FloatTensor)-1)-1
        rays = rays.squeeze()
        rgbs = rgbs.squeeze()
        scene_idx2 = batch['scene_idx2']
        # with torch.no_grad():
        #     self.list_log += [(view, scene_t)]
        #     new_log = (view, scene_t)
        #     count = self.list_log.count(new_log)
        #     count = count-1
            
        #     self.list1 = self.list[count: count+1024]
        # list = random.sample(range(0,230400),1024)
        list = self.list[scene_idx2*1024 : (scene_idx2+1)*1024]
        # list = random.sample(range(0,160000),1024)
        rays = rays[list]
        rgbs = rgbs[list]
        # rays = rays[self.list1]
        # rgbs = rgbs[self.list1]
        results = self(rays.float(), t_normalize = t_normalize)  #[1024,3]
        log['train/loss'] = loss = self.loss(results, rgbs)
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

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs, scene_t, t_num1 = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3) [160000, 8]
        rgbs = rgbs.squeeze() # (H*W, 3)
        # t_normalize = 2*scene_t/(t_num1-1)-1  #0
        t_normalize = 2*scene_t.type(torch.FloatTensor)/(t_num1.type(torch.FloatTensor)-1)-1
        # list = random.sample(range(0,160000),1024)
        # rays = rays[list]
        # rgbs = rgbs[list]
        # results = self(rays.float(), t_normalize = t_normalize)  #[160000,3]
        results = self(rays.float())
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        # img = results['rgb_coarse'].reshape(400,400,3).permute(2,0,1).cpu()
        img = results['rgb_coarse'].reshape(360,640,3).permute(2,0,1).cpu()
        # img1 = rgbs.reshape(400,400,3).permute(2,0,1).cpu()
        img1 = rgbs.reshape(360,640,3).permute(2,0,1).cpu()
        # img = img.cuda()
        # img1 = img1.cuda()
        # img = self.unpreprocess(img).squeeze().cpu()
        # img1 = self.unpreprocess(img1).squeeze().cpu()
        
        toPIL = transforms.ToPILImage()
        pic = toPIL(img)
        pic.save(f'random_{batch_nb}.jpg')
        pic = toPIL(img1)
        pic.save(f'random2_{batch_nb}.jpg')
        img1 = img1.unsqueeze(0)
        img = img.unsqueeze(0)
        img_vis = torch.cat((img1,img),dim=0).permute(2,0,3,1).reshape(img1.shape[2],-1,3).numpy()
        os.makedirs(f'/data1/liufengyi/get_results/nerfpl_t/runs_new1/{self.args.exp_name}/{self.args.exp_name}/',exist_ok=True)
        imageio.imwrite(f'/data1/liufengyi/get_results/nerfpl_t/runs_new1/{self.args.exp_name}/{self.args.exp_name}/{self.global_step:05d}_{self.idx:02d}.png', (img_vis*255).astype('uint8'))

        # if batch_nb == 0:
        #     W, H = self.args.img_wh
        #     img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
        #     img = img.permute(2, 0, 1) # (3, H, W)
        #     img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        #     depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        #     stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
        #     self.logger.experiment.add_images('val/GT_pred_depth',
        #                                        stack, self.global_step)
        self.idx += 1
        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        self.log('val/loss_mean', mean_loss.item(), prog_bar=True)
        self.log('val/psnr_mean', mean_psnr.item(), prog_bar=True)

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }

    def save_ckpt(self, name='final'):
        save_dir = f'/data1/liufengyi/get_results/nerfpl_t/runs_new1/{self.args.exp_name}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/{name}.tar'
        ckpt = {
            # 'global_step': self.global_step,
            # 'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
            # 'network_mvs_state_dict': self.MVSNet.state_dict()}
                'nerf_coarse_state_dict' :self.models[0].state_dict(),
                'nerf_fine_state_dict' :self.models[1].state_dict()}
        # if self.render_kwargs_train['network_fine'] is not None:
        #     ckpt['network_fine_state_dict'] = self.render_kwargs_train['network_fine'].state_dict()
        torch.save(ckpt, path)
        print('Saved checkpoints at', path)

if __name__ == '__main__':
    args = config_parser()
    system = NeRFSystem(args)
    a = os.path.join(f'/data1/liufengyi/get_results/nerfpl_t/runs_new1/{args.exp_name}/ckpts/','{epoch:02d}')
    dirpath = f'/data1/liufengyi/get_results/nerfpl_t/runs_new1/{args.exp_name}/ckpts/'
    filename = '{epoch:02d}-{val/loss:.2f}'
    checkpoint_callback = ModelCheckpoint(dirpath = dirpath,
                                          filename = filename,
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,)

    logger = TestTubeLogger(
        save_dir="/data1/liufengyi/get_results/nerfpl_t/logs",
        name=args.exp_name,
        debug=False,
        create_git_tag=False
    )
    trainer = Trainer(max_epochs=args.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      # callbacks=[checkpoint_callback],
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=args.num_gpus,
                      distributed_backend='ddp' if args.num_gpus > 1 else None,
                      num_sanity_val_steps=1,   #训练之前进行校验
                    #   check_val_every_n_epoch = max(system.args.num_epochs//system.args.N_vis,1),
                      check_val_every_n_epoch = 1,   #一个epoch校验一次
                      benchmark=True,
                      precision=16,  
                    #   val_check_interval=0.1,    #0.1个epoch校验一次
                      amp_level='O1',
                      log_every_n_steps = 1)   #每隔1次迭代记录一下logger

    trainer.fit(system)
    system.save_ckpt()
    torch.cuda.empty_cache()

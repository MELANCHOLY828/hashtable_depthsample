import torch
import os
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
from torchsearchsorted import searchsorted

from models.rendering import render_rays1,render_rays, render_rays2
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
import configargparse

torch.backends.cudnn.benchmark = True

def config_parser(cmd=None):
    parser = ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/data1/liufengyi/all_datasets/facebook/cook_spinach_img/resize_480*640/',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='facebook_NeRV',
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test_final',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[640, 480],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, default = "/data1/liufengyi/get_results/non_synchronized_NeRF/runs_new/non_synchronized_NeRF/ckpts/epoch=03-val_loss=0.004.ckpt", 
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    if cmd is not None:
            return parser.parse_args(cmd)
    else:
            return parser.parse_args(args=[])


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,
                      t_normalize):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays2( models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        t_normalize = t_normalize,
                        test_time=False
                        )

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = config_parser()
    w, h = args.img_wh
    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    # dataset = dataset_dict[args.dataset_name](**kwargs)
    val_dir = args.root_dir
    flag = 0            #标志test是有gt还是无gt  为0代表无gt
    dataset = dataset_dict[args.dataset_name](root_dir=val_dir, split='train1', max_len=-1)

    embedding_xyz = Embedding(3, 10)
    # embedding_xyz = Embedding(4, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    latentcode = LatentCode()
    # args.ckpt_path = "/data1/liufengyi/get_results/non_synchronized_NeRF/runs_new/non_synchronized_NeRF/ckpts/epoch=04-val_loss=0.004-v1.ckpt"
    if args.ckpt_path is not None and args.ckpt_path != 'None':
        ckpts = [args.ckpt_path]
        print('Found ckpts', ckpts)

    if len(ckpts) > 0 :
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    load_ckpt(latentcode, args.ckpt_path, model_name='LatentCode')
    # nerf_coarse.load_state_dict(ckpt['nerf_coarse_state_dict'])
    # nerf_fine.load_state_dict(ckpt['nerf_fine_state_dict'])

    # load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    # load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()
    latentcode.cuda().eval()
    models = [nerf_coarse, nerf_fine, latentcode]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []
    dir_name = f'/data1/liufengyi/get_results/non_synchronized_NeRF/results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    # for i in tqdm(range(len(dataset))):   #test 多视角 无gt
    #     sample = dataset[i]
    #     rays = sample['rays'].cuda()
    #     rays = rays.squeeze()
    #     # rgbs = sample['rgbs'].squeeze().cuda
    #     view = sample['view']
    #     extract_flamenum = sample['extract_flamenum']
    #     t_num1 = torch.tensor(30.0)
    #     for t in range(0,30):
    #         t = extract_flamenum/3 + t
    #         t_normalize = 2*t/t_num1-1
    #         results = batched_inference(models, embeddings, rays.float(),
    #                                 args.N_samples, args.N_importance, args.use_disp,
    #                                 args.chunk,
    #                                 dataset.white_back,
    #                                 t_normalize = t_normalize)
    #         img_pred1 = results['rgb_fine'].view(h, w, 3)
    #         img_pred = img_pred1.cpu().numpy()
    #         img_pred_ = (img_pred*255).astype(np.uint8)
    #         imgs += [img_pred_]
    #         time = int(t)
    #         imageio.imwrite(os.path.join(dir_name, f'mulview_view{i:03d}_flame{time:02d}.png'), img_pred_)

    #         if 'rgbs' in sample:
    #             rgbs = sample['rgbs']
    #             img_gt = rgbs.view(h, w, 3)
    #             img_gt = img_gt.unsqueeze(0)
    #             img_pred1 = img_pred1.unsqueeze(0)  #[1,h,w,3]
    #             img_cha = abs(img_gt - img_pred1.cpu())
    #             # img_cha = 1.-(img_gt - img_gt)
    #             img_vis = torch.cat((img_gt,img_pred1.cpu(),img_cha),dim=0).permute(1,0,2,3).reshape(img_gt.shape[1],-1,3).numpy()
                
    #             # imageio.imwrite(os.path.join(dir_name, f'liu_{i:03d}_compare.png'), (img_vis*255).astype(np.uint8))
    #             imageio.imwrite(os.path.join(dir_name, f'view_{i:03d}_compare.png'), (img_vis*255).astype(np.uint8))
    #             psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
        
    # for i in tqdm(range(len(dataset))):   #test 单视角 有gt
    #     sample = dataset[i]
    #     rays = sample['rays'].cuda()
    #     rays = rays.squeeze()
    #     # rgbs = sample['rgbs'].squeeze().cuda
    #     # view = sample['view']
    #     t_num1 = torch.tensor(30.0)
    #     image_t = sample['time']
    #     time = int(image_t)
    #    # image_t = image_t*torch.ones_like(rays[:, :1])
    #     # t_normalize = 2*scene_t/(t_num1-1)-1  #0
    #     t_normalize = 2*image_t.type(torch.FloatTensor)/(t_num1.type(torch.FloatTensor))-1
    #     results = batched_inference(models, embeddings, rays.float(),
    #                             args.N_samples, args.N_importance, args.use_disp,
    #                             args.chunk,
    #                             dataset.white_back,
    #                             t_normalize = t_normalize)
    #     img_pred1 = results['rgb_fine'].view(h, w, 3)
    #     img_pred = img_pred1.cpu().numpy()
    #     img_pred_ = (img_pred*255).astype(np.uint8)
    #     imgs += [img_pred_]
        
    #     imageio.imwrite(os.path.join(dir_name, f'view_{i:03d}_{time:02d}.png'), img_pred_)

    #     if 'rgbs' in sample:
    #         rgbs = sample['rgbs']
    #         img_gt = rgbs.view(h, w, 3)
    #         img_gt = img_gt.unsqueeze(0)
    #         img_pred1 = img_pred1.unsqueeze(0)  #[1,h,w,3]
    #         img_cha = abs(img_gt - img_pred1.cpu())
    #         # img_cha = 1.-(img_gt - img_gt)
    #         img_vis = torch.cat((img_gt,img_pred1.cpu(),img_cha),dim=0).permute(1,0,2,3).reshape(img_gt.shape[1],-1,3).numpy()
            
    #         imageio.imwrite(os.path.join(dir_name, f'view_{i:03d}_compare.png'), (img_vis*255).astype(np.uint8))
    #         psnrs += [metrics.psnr(img_gt, img_pred).item()]
    # imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)


    loss_path1 = '/data1/liufengyi/all_datasets/facebook/cook_spinach_img/resize_480*640/'

    for i in tqdm(range(len(dataset))):   #对train数据集进行测试 得到误差
        sample = dataset[i]
        # img = sample['rgbs']
        rays = sample['rays'].cuda()
        rays = rays.squeeze()
        # rgbs = sample['rgbs'].squeeze().cuda
        # view = sample['view']
        image_t = torch.tensor(sample['image_t'])
        view = sample['view']
        flame = sample['flame']
        loss_path = os.path.join(f'{loss_path1}', f'NeRF_{view:02d}_loss')
        if not os.path.exists(loss_path):
            os.mkdir(loss_path)
        # time = int(image_t)
       # image_t = image_t*torch.ones_like(rays[:, :1])
        # t_normalize = 2*scene_t/(t_num1-1)-1  #0
        # t_normalize = 2*image_t.type(torch.FloatTensor)/(t_num1.type(torch.FloatTensor))-1
        results = batched_inference(models, embeddings, rays.float(),
                                args.N_samples, args.N_importance, args.use_disp,
                                args.chunk,
                                dataset.white_back,
                                t_normalize = image_t)
        img_pred1 = results['rgb_fine'].view(h, w, 3)
        img_pred = img_pred1.cpu().numpy()
        # img_pred_ = (img_pred*255).astype(np.uint8)
        # imgs += [img_pred_]
        
        # imageio.imwrite(os.path.join(loss_path, f'view_{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            img_gt = img_gt.unsqueeze(0)
            img_pred1 = img_pred1.unsqueeze(0)  #[1,h,w,3]
            img_cha = abs(img_gt - img_pred1.cpu())
            # img_cha1 = img_cha*255
            img_cha1 = img_cha.squeeze()
            imgray = cv2.cvtColor(img_cha1.numpy(),cv2.COLOR_BGR2GRAY)
            imageio.imwrite(os.path.join(loss_path, f'image{flame:02d}.png'), (imgray*255).astype(np.uint8))
            
            # img_sample = (imgray/imgray.sum()).reshape(1, -1)
            # img_sample = np.cumsum(img_sample, -1)
            
            # sample_list = []
            # sample_rand = torch.rand(1, 1024*4)

            # inds = searchsorted(torch.tensor(img_sample).float(), sample_rand, side = 'right')
            # # for i in range(len(sample_rand)):
            # #     sample_list += abs(img_cha1 - sample_rand[i]).argmin()
            # img_cha1.reshape(-1,3)[inds] = 1
            
            # imageio.imwrite(os.path.join(loss_path, f'vis.png'), (img_cha1.numpy()*255).astype(np.uint8))
            
            # img_vis = torch.cat((img_gt,img_pred1.cpu(),img_cha),dim=0).permute(1,0,2,3).reshape(img_gt.shape[1],-1,3).numpy()
            



            # imageio.imwrite(os.path.join(loss_path, f'view_{i:03d}_compare.png'), (img_vis*255).astype(np.uint8))
            # psnrs += [metrics.psnr(img_gt, img_pred).item()]
    # imageio.mimsave(os.path.join(loss_path, f'{args.scene_name}.gif'), imgs, fps=30)







    
    
    # if psnrs:
    #     mean_psnr = np.mean(psnrs)
    #     print(f'Mean PSNR : {mean_psnr:.3f}')
    #     print('PSNR:',psnrs)
# a = torch.rand(10)
# a = a/a.sum()

# b = torch.rand(10)
# for i in range(len(a)):
#     if i == 0:
#         b[i] = a[i]
#     else:
#         b[i] = a[:i+1].sum()
    
# c = torch.rand(10)
# abs(b-c[0]).argmax()



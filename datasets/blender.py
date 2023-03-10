import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from torch.utils.data import DataLoader
class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(400, 400), depth = False, flag_depth = False, xyz_min=None, xyz_max=None):
        self.root_dir = root_dir
        self.split = split
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.flag_depth = flag_depth
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.depth = depth
        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train' and self.depth == False: # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
            self.xyz_max = -self.xyz_min
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)   #[3,4]

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1.-img[:, -1:]) # blend A to RGB
                if self.flag_depth:
                    a =list(frame['file_path'])
                    a.insert(-4,'_depth')
                    path = ''.join(a)
                    depth_path = os.path.join(self.root_dir, f"{path}.npz")
                    train_depth = np.load(depth_path)['arr_0']
                    train_depth = torch.from_numpy(train_depth).reshape(-1,1)
                    img = torch.cat((img, train_depth), dim = -1)
                else:
                    self.all_rgbs += [img]   # img [160000,3]   
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)  [160000, 3]
                pts_nf = torch.stack([rays_o+rays_d*self.near, rays_o+rays_d*self.far])
                self.xyz_min = torch.minimum(self.xyz_min, pts_nf.amin((0,1)))
                self.xyz_max = torch.maximum(self.xyz_max, pts_nf.amax((0,1)))
                
                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
    def get_box(self):
        return self.xyz_min, self.xyz_max
    
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train' and self.depth == False:
            return len(self.all_rays)  #100*160000
            # return 100
        if self.split == 'train' and self.depth == True:
            return len(self.meta['frames'])
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train' and self.depth == False: # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        elif self.split == 'train' and self.depth == True:
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}
        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = '/home/zhangruiqi/zrq_project/nerf-pytorch/data/nerf_synthetic/lego'
    dataset = BlenderDataset(root_dir = root_dir, split='train',flag_depth = True)
    train_dataset = DataLoader(dataset = dataset,
                            batch_size = 1024,
                            num_workers= 0,
                            shuffle=False)
    for i,sample in enumerate(train_dataset):
        print("liu")
        data = sample
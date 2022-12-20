import os
import glob
import random
import time
from kornia import create_meshgrid
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from torchvision import transforms
import random
def data_augmentation(images):
    mode = random.randint(0, 4)
    if mode == 0:
        # random brightness
        brightness_factor = 1.0 + random.uniform(-0.2, 0.3)
        xi = tf.adjust_brightness(images, brightness_factor)
    elif mode == 1:
        # random saturation
        saturation_factor = 1.0 + random.uniform(-0.2, 0.5)
        xi = tf.adjust_saturation(images, saturation_factor)
    elif mode == 2:
        # random hue
        hue_factor = random.uniform(-0.2, 0.2)
        xi = tf.adjust_hue(images, hue_factor)
    elif mode == 3:
        # random contrast
        contrast_factor = 1.0 + random.uniform(-0.2, 0.4)
        xi = tf.adjust_contrast(images, contrast_factor)
    return xi

def random_subsequence(seq, length, min_stride=1, max_stride=1):
    """Returns a random subsequence with min_stride <= stride <= max_stride.
    For example if self.length = 4 and we ask for a length 2
    sequence (with default min/max_stride=1), there are three possibilities:
    [0,1], [1,2], [2,3].
    Args:
        seq: list of image sequence indices
        length: the length of the subsequence to be returned.
        min_stride: the minimum stride (> 0) between elements of the sequence
        max_stride: the maximum stride (> 0) between elements of the sequence
    Returns:
        A random, uniformly chosen subsequence of the requested length
        and stride.
    """
    # First pick a stride.
    if max_stride == min_stride:
      stride = min_stride
    else:
      stride = np.random.randint(min_stride, max_stride+1)

    # Now pick the starting index.
    # If the subsequence starts at index i, then its final element will be at
    # index i + (length - 1) * stride, which must be less than the length of
    # the sequence. Therefore i must be less than maxval, where:
    maxval = len(seq) - (length - 1) * stride
    start = np.random.randint(0, maxval)
    end = start + 1 + (length - 1) * stride
    return seq[start:end:stride]

def pose2mat(pose):
    """Convert pose matrix (3x5) to extrinsic matrix (4x4) and
       intrinsic matrix (3x3)
    
    Args:
        pose: 3x5 pose matrix
    Returns:
        Extrinsic matrix (4x4) and intrinsic matrix (3x3)
    """
    extrinsic = torch.eye(4)
    extrinsic[:3, :] = pose[:, :4]
    inv_extrinsic = torch.inverse(extrinsic)
    extrinsic = torch.inverse(inv_extrinsic)
    h, w, focal_length = pose[:, 4]
    intrinsic = torch.Tensor([[focal_length, 0, w/2],
                              [0, focal_length, h/2],
                              [0,            0,   1]])

    return extrinsic, intrinsic

def convert_llff(pose):
    """Convert LLFF poses to PyTorch convention (w2c extrinsic and hwf)
    """
    hwf = pose[:3, 4:]

    ext = torch.eye(4)
    ext[:3, :4] = pose[:3, :4]
    mat = torch.inverse(ext)
    mat = mat[[1, 0, 2]]
    mat[2] = -mat[2]
    mat[:, 2] = -mat[:, 2]

    return torch.cat([mat, hwf], -1)




def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d




class MultiViewDataset2(Dataset):
    def __init__(self, root_dir, num_frames=3,
        img_hw=(360, 640), split='train',
        cam_indices=[],
        min_stride=1, max_stride=1,
        full_seq=False, random_reverse=True,
        num_cams=3, max_len=-1):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_hw = img_hw
        self.split = split
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.full_seq = full_seq # Only for testing
        self.num_cams = num_cams # numbers of cameras to load
        self.random_reverse = random_reverse # randomly reverse the image sequence
        self.cam_indices = cam_indices
        self.nc = 10 # 10 cameras in all dataset
        self.max_len = max_len
        # self.nfs = 0
        self.list_all = []
        # for i in range(0,1000):
        #     self.list_all.append(i)
        # self.list_train = random.sample(range(0,1000),800)
        # self.list_mid = set(self.list_all).difference(self.list_train)
        # self.list_mid = list(self.list_mid)
        # self.list_val = self.list_mid[:100]
        # self.list_test = self.list_mid[100:]

        for i in range(0,200):
            self.list_all.append(i)
        self.list_train = random.sample(range(0,200),180)
        self.list_mid = set(self.list_all).difference(self.list_train)
        self.list_mid = list(self.list_mid)
        self.list_val = self.list_mid[:10]
        self.list_test = self.list_mid[10:]

        self.white_back = False
        self.define_transforms()
        self._init_dataset()
        
        

    def _init_dataset(self):
        def proc_poses_bounds(input_poses):
            poses = input_poses[:, :-2].reshape([-1, 3, 5])
            bounds = input_poses[:, -2:]
            poses = [pose2mat(convert_llff(x)) for x in poses]  #poses[0]代表第一个视角下的内外参 poses[0][0]
            w2c = torch.stack([x[0] for x in poses])
            K = torch.stack([x[1] for x in poses])
            K[:, :2] *= self.img_hw[1] / K[:, 0:1, -1:] / 2 # Scale image accordingly
            return K, w2c, bounds

        self.metas = []
        # self.scene_paths = sorted(glob.glob(os.path.join(self.root_dir, '*', '*.h5')))
        self.scene_paths = sorted(glob.glob(os.path.join(self.root_dir, 'train', '*', 'scene3.h5')))
        self.frame_count = []
        for path in self.scene_paths:
            # scene = path[:-3]
            with h5py.File(path, 'r') as hf:
                self.num = hf['rgb'].shape[1]   #每个场景的帧数
                nfs = 20
                # nfs = 100                 #目前先用前100帧来进行测试
                multi_views = 10
                for multi_view in range(0,multi_views):
                    for nf in range(0,nfs):
                        self.metas +=  [(path,multi_view,nf)]
                #self.frame_count.append(nf)
        self.nfs = nfs
        if self.split == 'train':
            self.metas = [self.metas[i] for i in self.list_train]
        if self.split == 'val':
            self.metas = [self.metas[i] for i in self.list_val]
        if self.split == 'test':
            self.metas = [self.metas[i] for i in self.list_test]
        # Read poses
        loaded = []
        for path in self.scene_paths:
            pose_path = path[:-3] + '_pb.npy'
            loaded.append(torch.FloatTensor(np.load(pose_path)))
        poses_bounds = [proc_poses_bounds(x) for x in loaded]  #list [k,w2c,bd]
        
        # (#scenes, #cams, ...)
        self.K = torch.stack([x[0] for x in poses_bounds])
        self.w2c = torch.stack([x[1] for x in poses_bounds])
        self.bds = torch.stack([x[2] for x in poses_bounds])

    # def define_transforms(self):
    #     self.transforms = T.ToTensor()
    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])

    def __getitem__(self, scene_idx):

        sample = {}
        # scene_idx = scene_idx%800
        scene_idx = scene_idx%180
        scene_path, multi_view, nf = self.metas[scene_idx]
        with h5py.File(scene_path, 'r') as hf:
            tgt_rgb = hf['rgb'][multi_view, nf]
            
            # tgt_rgb = tgt_rgb.permute([2, 0, 1])
            tgt_rgb = self.transform(tgt_rgb)
            
        # toPIL = transforms.ToPILImage()
        # pic = toPIL(tgt_rgb)
        # pic.save('random.jpg')
        
        sample['tgt_rgb'] = tgt_rgb
        # sample['tgt_w2c'] = self.w2c[int(scene_idx//800), multi_view]
        # sample['tgt_K'] = self.K[int(scene_idx//800), multi_view]
        sample['tgt_w2c'] = self.w2c[int(scene_idx//180), multi_view]
        sample['tgt_K'] = self.K[int(scene_idx//180), multi_view]
        
        sample['bd'] = torch.FloatTensor([
            torch.min(self.bds[int(scene_idx//180)]) * .9,
            torch.max(self.bds[int(scene_idx//180)]) * 2.,
        ])
        near, far = sample['bd']
        sample['scene_t'] = nf
        sample['t_num'] = self.num
        sample['t_num1'] = self.nfs
        #后面根据nerf blender加的
        device = tgt_rgb.device
        w2c1=sample['tgt_w2c'].cpu()
        w2c1.numpy()
        c2w = np.linalg.inv(w2c1)
        c2w = torch.from_numpy(c2w)
        c2w = c2w.to(device)
        c2w = c2w[:3, :]
        h, w = self.img_hw
        focal = sample['tgt_K'][0,0]
        directions = get_ray_directions(h, w, focal)  #[360, 640, 3]
        rays_d , rays_o = get_rays(directions, c2w)   #[360*640, 3]
        sample['rays'] = torch.cat([rays_o, rays_d, near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1) # [360*640, 8]
        sample['rgbs'] = tgt_rgb.permute(1, 2, 0).reshape(-1, 3)   #[360*640, 3]
        return sample


        indices = [x for x in range(self.frame_count[scene_idx])]
        subseq = random_subsequence(indices, self.num_frames, self.min_stride, self.max_stride)

        if self.split == 'train' or self.split == 'valid':
            self.cam_indices = np.random.choice(self.nc, self.num_cams, replace=False) # 不可以取相同元素 First element is target element

        elif self.split == 'test':
            if not self.cam_indices:
                self.cam_indices = np.random.choice(self.nc, self.num_cams, replace=False)
            # subseq = [x for x in range(self.num_frames)] # TODO Might want to select a particular part?
            if self.full_seq:
                subseq = indices
            print(f'Current camera indices: {self.cam_indices}')
            # print(f'Subsequences: {subseq}')

        with h5py.File(self.scene_paths[scene_idx], 'r') as hf:
            tgt_idx = self.cam_indices[0]
            tgt_rgb = torch.FloatTensor(hf['rgb'][tgt_idx, subseq]) / 255.
            tgt_rgb = tgt_rgb.permute([0, 3, 1, 2])
            src_idx = self.cam_indices[1:]

            # Somehow faster than list comprehension
            tmp = []
            for i in src_idx:
                vol = torch.FloatTensor(hf['rgb'][i, subseq]).permute([0, 3, 1, 2])
                tmp.append(vol)
            src_rgb = torch.stack(tmp) / 255.

            # Foreground
            tgt_fg = torch.FloatTensor(hf['fg_rgb'][tgt_idx, subseq]).permute([0, 3, 1, 2]) / 255.

            # Background
            tmp = []
            for i in src_idx:
                tmp.append(torch.FloatTensor(hf['bg_rgb'][i]))
            src_bg = torch.stack(tmp).permute([0, 3, 1, 2]) / 255.
            tgt_bg = torch.FloatTensor(hf['bg_rgb'][tgt_idx]).permute([2, 0, 1]) / 255.
        

        # Randomly reverse sequences
        if bool(random.getrandbits(1)) and self.random_reverse:
            tgt_rgb = torch.flip(tgt_rgb, [0])
            src_rgb = torch.flip(src_rgb, [1])
            tgt_fg = torch.flip(tgt_fg, [0])
        
        # Pack data
        sample['tgt_rgb'] = tgt_rgb
        sample['src_rgb'] = src_rgb

        sample['src_w2c'] = self.w2c[scene_idx, src_idx]
        sample['src_K'] = self.K[scene_idx, src_idx]
        sample['tgt_w2c'] = self.w2c[scene_idx, tgt_idx]
        sample['tgt_K'] = self.K[scene_idx, tgt_idx]
        
        sample['bd'] = torch.FloatTensor([
            torch.min(self.bds[scene_idx]) * .9,
            torch.max(self.bds[scene_idx]) * 2.,
        ])

        sample['src_bg'] = src_bg
        sample['tgt_bg'] = tgt_bg
        sample['tgt_fg'] = tgt_fg
        tgt_mask = (tgt_fg[:, -1] > 0.5).float()
        sample['tgt_mask'] = tgt_mask

        return sample

    def __len__(self):
        return len(self.metas) * 100 if self.max_len<0 else 10
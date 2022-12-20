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
from .ray_utils import *
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

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg



def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def convert_llff1(pose):
    """Convert LLFF poses to PyTorch convention (w2c extrinsic and hwf)
    """
    hwf = pose[:3, 4:] 
    ###根据llff加的
    poses = pose.numpy()
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    poses, pose_avg = center_poses(poses)
    distances_from_center = np.linalg.norm(poses[..., 3], axis=1)
    val_idx = np.argmin(distances_from_center)

    ext = torch.eye(4)
    ext[:3, :4] = pose[:3, :4]
    mat = torch.inverse(ext)
    mat = mat[[1, 0, 2]]
    mat[2] = -mat[2]
    mat[:, 2] = -mat[:, 2]

    return torch.cat([mat, hwf], -1)

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
    rays_d = directions @ (c2w[:, :3].T).type(torch.FloatTensor) # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)

class MultiViewDataset3(Dataset):
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
        self.h, self.w = self.img_hw
        self.img_wh = [self.w, self.h]
        self.spheric_poses = False
        
        # for i in range(0,1000):
        #     self.list_all.append(i)
        # self.list_train = random.sample(range(0,1000),800)
        # self.list_mid = set(self.list_all).difference(self.list_train)
        # self.list_mid = list(self.list_mid)
        # self.list_val = self.list_mid[:100]
        # self.list_test = self.list_mid[100:]


        # f=open("/data1/liufengyi/all_datasets/list_nerft/val.txt","w+")
        # for i in list_val:
        #     f.write(str(i))
        #     f.write('\n')
        # f.close()


        if self.split == 'train':
            self.list_train = []
            f=open("/data1/liufengyi/all_datasets/list/list_nerft/train.txt","r")
            for line in f:
                self.list_train.append(int(line.strip('\n').split(',')[0]))

        if self.split == 'val':
            self.list_val = []
            f=open("/data1/liufengyi/all_datasets/list/list_nerft/val.txt","r")
            for line in f:
                self.list_val.append(int(line.strip('\n').split(',')[0]))

        # if self.split == 'test':
        #     self.list_test = []
        #     f=open("/data1/liufengyi/all_datasets/list_nerft/test.txt","r")
        #     for line in f:
        #         self.list_test.append(int(line.strip('\n').split(',')[0]))

        # for i in range(0,200):
        #     self.list_all.append(i)
        # self.list_train = random.sample(range(0,200),180)
        # self.list_mid = set(self.list_all).difference(self.list_train)
        # self.list_mid = list(self.list_mid)
        # self.list_val = self.list_mid[:10]
        # self.list_test = self.list_mid[10:]

        self.white_back = False
        self.define_transforms()
        self._init_dataset()
        
        

    def _init_dataset(self):
        def proc_poses_bounds(input_poses, img_wh):  #[10, 17]
            poses = input_poses[:, :-2].reshape([-1, 3, 5]) 
            bounds = input_poses[:, -2:]
            #自己加的
            hwf = poses[:, :, -1] #[10, 3]
            H, W = hwf[0, :2]
            focal = hwf[:, -1]  #[10]
            focal *= self.img_wh[0]/W   #[10]
            poses = poses.numpy()
            poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
            poses, pose_avg = center_poses(poses)
            distances_from_center = np.linalg.norm(poses[..., 3], axis=1)
            val_idx = np.argmin(distances_from_center)
            near_original = bounds.min()
            scale_factor = near_original*0.75
            bounds /= scale_factor
            poses[..., 3] /= scale_factor    #平移矩阵   #(10, 3, 4)
            self.poses = poses
            poses = torch.from_numpy(poses)
            
            return poses, focal, bounds   #numpy(10, 3, 4)  tensor[10]   tensor[10, 2]




            poses = [pose2mat(convert_llff(x)) for x in poses]  #poses[0]代表第一个视角下的内外参 poses[0][0]
            w2c = torch.stack([x[0] for x in poses])
            K = torch.stack([x[1] for x in poses])
            K[:, :2] *= self.img_hw[1] / K[:, 0:1, -1:] / 2 # Scale image accordingly
            return K, w2c, bounds


        def proc_poses_bounds1(input_poses):  #[10, 17]
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
                # nfs = 20
                nfs = 1                 #目前先用前100帧来进行测试
                multi_views = 9
                for multi_view1 in range(0,multi_views):   #9view进行train 和 val
                    for nf in range(0,nfs):
                        self.metas +=  [(path,multi_view1,nf)]
                #self.frame_count.append(nf)
        nfs = 100
        self.nfs = nfs
        # if self.split == 'train':
        #     self.metas = [self.metas[i] for i in self.list_train]
        # if self.split == 'val':
        #     self.metas = [self.metas[i] for i in self.list_val]
        if self.split == 'test':
            self.metas = []
            for nf in range(0,nfs):
                multi_view = random.randint(0,9)
                self.metas +=  [(path,multi_view,nf)]
            # self.metas = [self.metas[i] for i in self.list_test]
        # Read poses
        loaded = []
        for path in self.scene_paths:
            pose_path = path[:-3] + '_pb.npy'
            loaded.append(torch.FloatTensor(np.load(pose_path)))
        poses_bounds = [proc_poses_bounds(x, self.img_wh) for x in loaded]  #list [k,w2c,bd]
        
        # (#scenes, #cams, ...)
        self.c2w = torch.stack([x[0] for x in poses_bounds])   #tensor [1, 10, 3, 4]
        self.focal = torch.stack([x[1] for x in poses_bounds])   #[1, 10]
        self.bounds = torch.stack([x[2] for x in poses_bounds])  #[1, 10, 2]

        # self.K = torch.stack([x[0] for x in poses_bounds])
        # self.w2c = torch.stack([x[1] for x in poses_bounds])
        # self.bds = torch.stack([x[2] for x in poses_bounds])

    # def define_transforms(self):
    #     self.transforms = T.ToTensor()
    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                                    #             std=[0.229, 0.224, 0.225]),
                                    ])

    def __getitem__(self, scene_idx):

        sample = {}
        if self.split == 'train':

            scene_idx1 = scene_idx//100   #第几张图片 0-799
            scene_idx2 = scene_idx%100    #第几个1024
        elif self.split == 'val':
            scene_idx1 = scene_idx
            scene_idx2 = 0
        elif self.split == 'test':
            scene_idx1 = scene_idx
            scene_idx2 = 0
        scene_idx = scene_idx%800
        scene_path, multi_view, nf = self.metas[scene_idx1]
        if self.split == 'test':
            c2w = self.c2w[0, multi_view]
            focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
            radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
            self.poses_test = create_spiral_poses(radii, focus_depth)   
            focal = self.focal[0, multi_view]    
            c2w = torch.from_numpy(self.poses_test[scene_idx])
            
        else:
            # scene_idx = scene_idx%180
            
            with h5py.File(scene_path, 'r') as hf:
                tgt_rgb = hf['rgb'][multi_view, nf]
                
                # tgt_rgb = tgt_rgb.permute([2, 0, 1])
                tgt_rgb = self.transform(tgt_rgb)
                
            # toPIL = transforms.ToPILImage()
            # pic = toPIL(tgt_rgb)
            # pic.save('random.jpg')
            
            ####根据llff加的
            # focal = self.focal[int(scene_idx//800), multi_view]
            # c2w = self.c2w[int(scene_idx//800), multi_view]
            focal = self.focal[0, multi_view]
            c2w = self.c2w[0, multi_view]
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], focal) # (H, W, 3)
        rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
        if not self.spheric_poses:
            near, far = 0, 1
            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                    focal, 1.0, rays_o, rays_d)
            # near plane is always at 1.0
            # near and far in NDC are always 0 and 1
            # See https://github.com/bmild/nerf/issues/34
        else:
            # near = self.bounds.min()
            # far = min(8 * near, self.bounds.max())  # focus on central object only
            near = torch.min(self.bounds[int(scene_idx//800)]) * 0.9
                        
            far = torch.max(self.bounds[int(scene_idx//800)]) * 2  # focus on central object only
        


        # sample['tgt_rgb'] = tgt_rgb
        # # sample['tgt_w2c'] = self.w2c[int(scene_idx//800), multi_view]
        # # sample['tgt_K'] = self.K[int(scene_idx//800), multi_view]
        # sample['tgt_w2c'] = self.w2c[int(scene_idx//180), multi_view]
        # sample['tgt_K'] = self.K[int(scene_idx//180), multi_view]
        
        # sample['bd'] = torch.FloatTensor([
        #     torch.min(self.bds[int(scene_idx//180)]) * .9,
        #     torch.max(self.bds[int(scene_idx//180)]) * 2.,
        # ])
        # near, far = sample['bd']
        sample['scene_t'] = nf
        sample['t_num'] = self.num
        sample['t_num1'] = self.nfs
        sample['view'] = multi_view
        sample['scene_idx2'] = scene_idx2
        # #后面根据nerf blender加的
        # device = tgt_rgb.device
        # w2c1=sample['tgt_w2c'].cpu()
        # w2c1.numpy()
        # c2w = np.linalg.inv(w2c1)
        # c2w = torch.from_numpy(c2w)
        # c2w = c2w.to(device)
        # c2w = c2w[:3, :]
        # h, w = self.img_hw
        # focal = sample['tgt_K'][0,0]
        # directions = get_ray_directions(h, w, focal)  #[360, 640, 3]
        # rays_d , rays_o = get_rays(directions, c2w)   #[360*640, 3]
        sample['rays'] = torch.cat([rays_o, rays_d, near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1) # [360*640, 8]
        if self.split != 'test':
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
        # return len(self.metas) * 100 if self.max_len<0 else 10
        return len(self.metas) * 100 if self.max_len<0 else 100   #test
        # return len(self.metas) * 100 if self.max_len<0 else 1   #1帧
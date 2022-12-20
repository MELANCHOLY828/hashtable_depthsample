import os, sys
parentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,parentdir) 
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
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
from PIL import Image
from torch.utils.data import DataLoader
from ray_utils import *
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
def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)

class FacebookDataset(Dataset):
    def __init__(self, root_dir,
        img_hw=(2028, 2704), split='train',
        cam_indices=[],
        min_stride=1, max_stride=1, max_len=-1):
        self.root_dir = root_dir
        self.img_hw = img_hw
        self.split = split
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.max_len = max_len
        self.h, self.w = self.img_hw
        self.img_wh = [self.w, self.h]
        self.spheric_poses = False
        self.white_back = False
        self.define_transforms()    #ToTensor
        self._init_dataset()
        
        

    def _init_dataset(self):
        self.metas = []
        # self.scene_paths = sorted(glob.glob(os.path.join(self.root_dir, '*', '*.h5')))
        self.scene_paths = sorted(glob.glob(os.path.join(self.root_dir, 'cam*')))
        if self.split == 'train':
            self.scene_paths = self.scene_paths[:19]
        elif self.split == 'val':
            self.scene_paths = self.scene_paths[19]
        elif self.split == 'test':
            self.scene_paths = self.scene_paths[20]

        self.frame_count = []
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_views, 17)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_views, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_views, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
        assert H*self.img_wh[0] == W*self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        self.focal *= self.img_wh[0]/W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_views, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            self.image_time = []
            flag = 1000
            for view, scene_path in enumerate(self.scene_paths):
                print(view)
                extract_flamenum = view % 3
                image_lists = sorted(glob.glob(os.path.join(scene_path, '*.jpg')))
                c2w = torch.FloatTensor(self.poses[view])
                for image_t, image_path in enumerate(image_lists):
                    image_t += extract_flamenum/3 
                    img = Image.open(image_path).convert('RGB')
                    assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                        f'''{image_path} has different aspect ratio than img_wh, 
                            please check your data!'''
                    # img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img) # (3, h, w)
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    self.all_rgbs += [img]     #(h*w, 3)
                    self.image_time += [image_t]  #
                    if flag != view:
                        rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                        if not self.spheric_poses:
                            near, far = 0, 1
                            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                        self.focal, 1.0, rays_o, rays_d)
                                            # near plane is always at 1.0
                                            # near and far in NDC are always 0 and 1
                                            # See https://github.com/bmild/nerf/issues/34
                        else:
                            near = self.bounds.min()
                            far = min(8 * near, self.bounds.max()) # focus on central object only

                    self.all_rays += [torch.cat([rays_o, rays_d, 
                                                near * torch.ones_like(rays_o[:, :1]),
                                                far * torch.ones_like(rays_o[:, :1]),
                                                image_t * torch.ones_like(rays_o[:, :1])],
                                                1)] # (h*w, 8)
                    flag = view               
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 9)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split == 'val':
            print('val image is', self.scene_paths)
            self.c2w_val = self.poses[19]
            self.image_path_val = self.scene_paths

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                                    #             std=[0.229, 0.224, 0.225]),
                                    ])

    def __getitem__(self, idx):
        print("ok")
        sample = {}
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'time': None}
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w}
            image_lists = sorted(glob.glob(os.path.join(self.scene_paths, '*.jpg')))
            extract_flamenum = 19 % 3
            image_t = idx + extract_flamenum/3
            image_path = image_lists[idx]
            img = Image.open(image_path).convert('RGB')
            assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                f'''{image_path} has different aspect ratio than img_wh, 
                    please check your data!'''
            # img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img
            sample['time'] = image_t
        return sample


    def __len__(self):
        # return len(self.metas) * 100 if self.max_len<0 else 10
        return len(self.all_rays) if self.max_len<0 else 100   #test
        # return len(self.metas) * 100 if self.max_len<0 else 1   #1帧
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = '/data1/liufengyi/all_datasets/facebook/cook_spinach_img/extract_frame/'
    dataset = FacebookDataset(root_dir = root_dir, split='train', max_len=1)
    train_dataset = DataLoader(dataset = dataset,
                            batch_size = 1024,
                            num_workers= 0,
                            shuffle=False)
    for i,sample in enumerate(train_dataset):
        print("liu")
        data = sample

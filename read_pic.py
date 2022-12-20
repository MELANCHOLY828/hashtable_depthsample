import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import imageio
from datasets import dataset_dict
from torchvision import transforms
# dataset_name = "mvcam_change"
# dataset_name = "llff"
# dataset_name = "blenderl"
dataset_name = "facebook_dataset"
# dataset_name = "mvcam_pic"
dataset = dataset_dict[dataset_name]
# root_dir = "/data1/liufengyi/all_datasets/multi-view"
root_dir = "/data1/liufengyi/all_datasets/facebook/cook_spinach_img/extract_frame/"
# root_dir = "/home/liufengyi/test/nerf_pl-master/data/nerf_llff_data1/nerf_llff_data/horns"
# img_wh = (504, 378) 
img_hw = (360, 640)
# img_hw = [400, 400] 
img_wh = (640, 360)
# train_dataset = dataset(root_dir, img_wh=img_wh,
#                         spheric_poses = False, val_num = 1)
train_dataset = dataset(root_dir, split = 'train')                    
train_dataset1 = DataLoader(dataset = train_dataset,
                            batch_size = 1,
                            num_workers= 0,
                            shuffle=False)
def unpreprocess(data, shape=(1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

for i,sample in enumerate(train_dataset1):
    data = sample
    tgt_rgb = sample['tgt_rgb']
    bg_rgb = sample['bg_rgb']
    fg_rgb = sample['fg_rgb']
    toPIL = transforms.ToPILImage()
    pic = toPIL(tgt_rgb[0])
    pic.save('random.jpg')
    pic1 = toPIL(bg_rgb[0])
    pic1.save('bg_rgb.jpg')
    pic2 = toPIL(fg_rgb[0])
    pic2.save('fg_rgb.jpg')
    # scene_t = sample['scene_t']
    # t_num1 = sample['t_num1']
    # t_normalize = 2*scene_t/(t_num1-1)-1
    

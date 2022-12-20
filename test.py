# # import torch
# # from torch import nn
# # from torch.nn import functional as F
# # from torch.utils.data import DataLoader
# # import imageio
# # from datasets import dataset_dict
# # from torchvision import transforms
# # dataset_name = "mvcam_llff"
# # # dataset_name = "blenderl"
# # dataset = dataset_dict[dataset_name]
# # root_dir = "/data1/liufengyi/all_datasets/multi-view"
# # # root_dir = "/data1/liufengyi/mvsnerf_t/mvsnerf/nerf_synthetic/nerf_synthetic/lego"
# # img_hw = (360, 640)
# # # img_hw = [400, 400] 
# # train_dataset = dataset(root_dir, img_hw=img_hw,
# #                         num_frames=1, min_stride=25, max_stride=25)
# # # train_dataset = dataset(root_dir, img_wh=img_hw)                    
# # train_dataset1 = DataLoader(dataset = train_dataset,
# #                             batch_size = 1,
# #                             num_workers= 0,
# #                             shuffle=False)
# # def unpreprocess(data, shape=(1,3,1,1)):
# #         # to unnormalize image for visualization
# #         # data N V C H W
# #         device = data.device
# #         mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
# #         std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

# #         return (data - mean) / std

# # for i,sample in enumerate(train_dataset1):
# #     data = sample
# #     tgt_rgb = sample['tgt_rgb']
# #     tgt_rgb = unpreprocess(tgt_rgb)
# #     toPIL = transforms.ToPILImage()
# #     pic = toPIL(tgt_rgb[0])
# #     pic.save('random.jpg')
# #     # scene_t = sample['scene_t']
# #     # t_num1 = sample['t_num1']
# #     # t_normalize = 2*scene_t/(t_num1-1)-1
    

# # # view_list = []
# # # time_list = []
# # # all_list = []
# # # f=open("/data1/liufengyi/all_datasets/list_nerft/train.txt","r")
# # # for line in f:
# # #         num = int(line.strip('\n').split(',')[0])
# # #         view = num//100
# # #         time = num%100
# # #         all_list.append((view,time))
# # #         view_list.append(view)
# # #         time_list.append(time)

# # # from xlsxwriter.workbook import Workbook
# # # import xlwt
# # # import xlrd
# # # from xlutils.copy import copy
# # # workbook = Workbook(r'test1.xlsx') # 创建xlsx
# # # worksheet = workbook.add_worksheet('A') # 添加sheet
# # # red = workbook.add_format({'color':'red'}) # 颜色对象
# # # styleBlueBkg = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
# # # for i in range(0,9):
# # #         for j in range(0,100):
# # #                 if (i,j) in all_list:
# # #                         print('liu')
# # #                         ws.write(i,col,ro.cell(i, col).value,styleBlueBkg)

# # #                         # worksheet.write_rich_string(i, j, "ok")
# # #                         worksheet.write(j,i,'train')
# # # # worksheet.write(0, 0, 'sentences') # 0，0表示row，column，sentences表示要写入的字符串
# # # # test_list = ["我爱", "中国", "天安门"]
# # # # test_list.insert(1, red) # 将颜色对象放入需要设置颜色的词语前面
# # # # print(test_list)
# # # # worksheet.write_rich_string(1, 0, *test_list) # 写入工作簿
# # # workbook.close() # 记得关闭
# import torch
# import os
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm
# import imageio
# import cv2
# from torchvision import transforms as T
# import colormap
# transform = T.Compose([T.ToTensor(),
#                                     # T.Normalize(mean=[0.485, 0.456, 0.406],
#                                     #             std=[0.229, 0.224, 0.225]),
#                                     ])
# img_cv = cv2.imread('/data1/liufengyi/get_results/nerfpl_t/runs_new/mvcam_final/mvcam_final/79999_03.png')
# img_cv1 = transform(img_cv[:,:640,:]).unsqueeze(0).permute(0,2,3,1)
# # img_cv1 = torch.from_numpy(img_cv[:,:640,:]).unsqueeze(0)
# img_cv2 = transform(img_cv[:,640:,:]).unsqueeze(0).permute(0,2,3,1)
# # img_cv2 = torch.from_numpy(img_cv[:,640:,:]).unsqueeze(0)
# img_cha = abs(img_cv1-img_cv2)
# img_vis = torch.cat((img_cv1,img_cv2,img_cha),dim=0).permute(1,0,2,3).reshape(img_cv1.shape[1],-1,3).numpy()
# img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
# imageio.imwrite(os.path.join('/data1/liufengyi/get_results/nerfpl_t/runs_new/mvcam_final/mvcam_final/', '03_com.png'), (img_vis*255).astype(np.uint8))

# # cv2.imwrite('/data1/liufengyi/get_results/nerfpl_t/runs_new/mvcam_final/mvcam_final/', '160_com.png'), img_vis.astype(np.uint8)


# import cv2 as cv


# img = cv.imread("../images/test.jpg")

# cv.imshow("test", img)

# dsc = cv.applyColorMap(img, cv.COLORMAP_COOL)

# # cv.imshow("COOL", dsc)

# img1 = cv.imread('/data1/liufengyi/get_results/nerfpl_t/runs_new/mvcam_final/mvcam_final/79999_03.png')

# color_image = cv.applyColorMap(img1, cv.COLORMAP_JET)

# cv.imshow("JET", color_image)

# cv.imshow("canjian", img1)



import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays1,render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
import configargparse
dataset_name = 'mvcam_llff1'
scene_name = 'test_final'
dir_name = f'/data1/liufengyi/get_results/nerfpl_t/results_1/{dataset_name}/{scene_name}'
img = []
for i in range(0,10):

    img += [imageio.imread(os.path.join(dir_name, f'liu_00{i}.png'))]
    
for i in range(10,100):
    img += [imageio.imread(os.path.join(dir_name, f'liu_0{i}.png'))]
imageio.mimsave(os.path.join(dir_name, f'1_{scene_name}.gif'), img[::5], fps=1)
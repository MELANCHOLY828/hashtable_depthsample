from .blender import BlenderDataset
from .llff import LLFFDataset
from .blender1 import BlenderDataset1
# from .mvcam_change import MultiViewDataset2
# from .mvcam_llff import MultiViewDataset1
# from .mvcam_llff1 import MultiViewDataset3
# from .mvcam_pic import MultiViewDataset_pic
# from .facebook_data import FacebookDataset
# from .facebook_resize import FacebookDataset1
# from .facebook_NeRV import FacebookDataset2
# from .facebook_NeRV1 import FacebookDataset3
# from .facebook_grid import FacebookGrid
# from .Blender_VoxGo import *
dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'blender1':BlenderDataset1,}
                # 'mvcam_change': MultiViewDataset2,
                # 'mvcam_llff': MultiViewDataset1,
                # 'mvcam_llff1': MultiViewDataset3,
                # "mvcam_pic": MultiViewDataset_pic,
                # "facebook_dataset": FacebookDataset,
                # "facebook_resize": FacebookDataset1,
                # "facebook_NeRV": FacebookDataset2,
                # "facebook_NeRV1": FacebookDataset3,
                # "facebook_grid": FacebookGrid}
import os, sys
from torch import nn
from kornia import create_meshgrid
import time
# from macpath import split
parentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,parentdir) 
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from pyexpat import model
import torch
from torchsearchsorted import searchsorted
import torch.nn.functional as F
from sh import *
# from .torchsearchsorted.src.torchsearchsorted import searchsorted
BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')
__all__ = ['render_rays']

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = searchsorted(cdf.float(), u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

def render_rays1(models,     
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                
                test_time=False,
                t_normalize = 0    #每个像素的时间都不同
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(model, model_latent, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]   #[32768, 64, 3]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        # t_normalize = t_normalize.cuda().unsqueeze(2).view(-1, 1)
        t_normalize = t_normalize.cuda().view(-1, 1)
        if not weights_only:   #[32768, 27]
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, 27)  

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]   #32768 * 64   train 1024 * 64
        out_chunks = []
        for i in range(0, B, chunk):   #64次  每次像素点都选上了 每次是一个深度，重复64次
            # Embed positions by chunk
            xyz_1 = xyz_[i:i+chunk]
            # t_normalize1 = torch.ones_like(xyz_1[:, 0:1]) * t_normalize.cuda()
            t_normalize1 = t_normalize[i:i+chunk]
            # xyzt_ = torch.cat([xyz_1, t_normalize1], -1)
            xyz_embedded = embedding_xyz(xyz_1)
            t_latent = model_latent(t_normalize1)     
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded, 
                                             t_latent, dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]   #每个list [1024*32,4] 64个list

        out = torch.cat(out_chunks, 0) #[2097152, 4]
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)  # [32768, 64, 4]
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights


    # Extract models from lists
    model_latent = models[2]
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]   #32768
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, 27)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)   64个depth
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)  #[32768, 64]   32768个像素点
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)  [32768, 64, 3]
    # t_normalize = t_normalize.unsqueeze(2).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    if test_time:
        t_normalize_1 = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
        weights_coarse = \
            inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True, t_normalize = t_normalize_1)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
        rgb_coarse, depth_coarse, weights_coarse = \
            inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize_coarse)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                 }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        t_normalize_fine = t_normalize.unsqueeze(1).expand(xyz_fine_sampled.shape)[:,:,-2:-1]
        rgb_fine, depth_fine, weights_fine = \
            inference(model_fine, model_latent, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize_fine)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result


def render_rays2(models,     
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                
                test_time=False,
                t_normalize = 0    #每个像素的时间相同
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(model, model_latent, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]   #[32768, 64, 3]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        # t_normalize = t_normalize.cuda().unsqueeze(2).view(-1, 1)
        t_normalize = t_normalize.cuda().view(-1, 1)
        t_latent = model_latent(t_normalize)
        if not weights_only:   #[32768, 27]
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, 27)  

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]   #32768 * 64   train 1024 * 64
        out_chunks = []
        for i in range(0, B, chunk):   #64次  每次像素点都选上了 每次是一个深度，重复64次
            # Embed positions by chunk
            xyz_1 = xyz_[i:i+chunk]
            # t_normalize1 = torch.ones_like(xyz_1[:, 0:1]) * t_normalize.cuda()
            # t_normalize1 = t_normalize[i:i+chunk]
            # t_latent_1 = t_latent[i:i+chunk]
            # xyzt_ = torch.cat([xyz_1, t_normalize1], -1)
            xyz_embedded = embedding_xyz(xyz_1)
            # t_latent = model_latent(t_normalize1)     
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded, 
                                             t_latent.expand(xyz_embedded.shape[0], t_latent.shape[-1]), dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]   #每个list [1024*32,4] 64个list

        out = torch.cat(out_chunks, 0) #[2097152, 4]
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)  # [32768, 64, 4]
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights


    # Extract models from lists
    model_latent = models[2]
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]   #32768
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, 27)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)   64个depth
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)  #[32768, 64]   32768个像素点
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)  [32768, 64, 3]
    # t_normalize = t_normalize.unsqueeze(2).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    if test_time:
        weights_coarse = \
            inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True, t_normalize = t_normalize)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
        rgb_coarse, depth_coarse, weights_coarse = \
            inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                 }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        # t_normalize_fine = t_normalize.unsqueeze(1).expand(xyz_fine_sampled.shape)[:,:,-2:-1]
        rgb_fine, depth_fine, weights_fine = \
            inference(model_fine, model_latent, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights


    # Extract models from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    if test_time:
        weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                 }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result



def render_grid(models,     
                embeddings,
                rays,
                world_size,
                grid_bounds,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                
                test_time=False,
                t_normalize = 0    #每个像素的时间相同
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    
    
    def rendering(model_MLP_dir, sigama_coarse, feature_coarse, dir_,
                        dir_embedded, z_vals, weights_only=False):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        sh_deg = 2
        N_samples_ = feature_coarse.shape[1]   #[32768, 64, 3]
        # Embed directions
        feature_coarse = feature_coarse.view(-1, feature_coarse.shape[-1]) # (N_rays*N_samples_, 3)
        
        
        if not weights_only:   #[32768, 27]
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, 27)  
            # dir_1 = torch.repeat_interleave(dir_, repeats=N_samples_, dim=0)
        feature_dir = torch.cat([feature_coarse, dir_embedded], 1)
        # rgb = model_MLP_dir(feature_dir)
        # return rgb, rgb, rgb
        # Perform model inference to get rgb and raw sigma
        B = feature_coarse.shape[0]   #32768 * 64   train 1024 * 64
        out_chunks = []
        for i in range(0, B, chunk):   #64次  每次像素点都选上了 每次是一个深度，重复64次
            # Embed positions by chunk
            
            feature_dir_ = feature_dir[i:i+chunk]
            out_chunks += [model_MLP_dir(feature_dir_)]   #每个list [1024*32,4] 64个list
        # rgb = eval_sh(sh_deg, feature_coarse.reshape(
        #         *feature_coarse.shape[:-1],-1, (sh_deg + 1) ** 2), dir_1)
        rgb = torch.cat(out_chunks, 0) #[2097152, 4]
        sigama = sigama_coarse.view(N_rays, N_samples_)
        rgbs= rgb.view(N_rays, N_samples_, 3)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigama.shape, device=sigama.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigama+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights


    # Extract models from lists
    model_HashSiren = models[0]
    model_MLP_dir = models[1]
    
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    xyz_min = grid_bounds[0].squeeze()
    xyz_max = grid_bounds[1].squeeze()
    
    # Decompose the inputs
    N_rays = rays.shape[0]   #32768
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, 27)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)   64个depth
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)  #[32768, 64]   32768个像素点
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)  [32768, 64, 3]
                         
                        
    # xyz_coarse_norm = ((xyz_coarse_sampled - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1
    xyz_coarse_norm = ((xyz_coarse_sampled - xyz_min) / (xyz_max - xyz_min)) * 2 - 1
    output_feature = model_HashSiren(rays_o)
    
    
    output_feature = output_feature.reshape(world_size[2], world_size[1], world_size[0], 28).permute(3,0,1,2)
    sigama = output_feature[0:1]
    feature_ = output_feature[1:]
    
    # #加一个激活函数
    # activate_f = nn.Sigmoid()
    # feature_ = activate_f(feature_)
    
    
    
    sigama_coarse = F.grid_sample(sigama.unsqueeze(0), xyz_coarse_norm.view(1,1,*xyz_coarse_norm.shape), mode='bilinear', align_corners=True).squeeze().unsqueeze(-1)
    feature_coarse = F.grid_sample(feature_.unsqueeze(0), xyz_coarse_norm.view(1,1,*xyz_coarse_norm.shape), mode='bilinear', align_corners=True).squeeze().permute(1,2,0)
    
    # if test_time:
    #     weights_coarse = \
    #         rendering(sigama_coarse, feature_coarse, rays_d,
    #                     dir_embedded, z_vals, weights_only=True)
    #     result = {'opacity_coarse': weights_coarse.sum(1)}
    # else:
    #     # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    #     rgb_coarse, depth_coarse, weights_coarse = \
    #         rendering(sigama_coarse, feature_coarse, rays_d,
    #                     dir_embedded, z_vals, weights_only=False)
    if test_time:
        weights_coarse = \
            rendering(model_MLP_dir, sigama_coarse, feature_coarse, rays_d,
                        dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
        rgb_coarse, depth_coarse, weights_coarse = \
            rendering(model_MLP_dir, sigama_coarse, feature_coarse, rays_d,
                        dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                    'depth_coarse': depth_coarse,
                    'opacity_coarse': weights_coarse.sum(1)
                    }
    
    
    
    # t_normalize = t_normalize.unsqueeze(2).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    # if test_time:
    #     weights_coarse = \
    #         inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
    #                   dir_embedded, z_vals, weights_only=True, t_normalize = t_normalize)
    #     result = {'opacity_coarse': weights_coarse.sum(1)}
    # else:
    #     # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    #     rgb_coarse, depth_coarse, weights_coarse = \
    #         inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
    #                   dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize)
    #     result = {'rgb_coarse': rgb_coarse,
    #               'depth_coarse': depth_coarse,
    #               'opacity_coarse': weights_coarse.sum(1)
    #              }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        xyz_fine_norm = ((xyz_fine_sampled - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1
        
        sigama_fine = F.grid_sample(sigama.unsqueeze(0), xyz_fine_norm.view(1,1,*xyz_fine_norm.shape), mode='bilinear', align_corners=True).squeeze().unsqueeze(-1)
        feature_fine = F.grid_sample(feature_.unsqueeze(0), xyz_fine_norm.view(1,1,*xyz_fine_norm.shape), mode='bilinear', align_corners=True).squeeze().permute(1,2,0)
        # t_normalize_fine = t_normalize.unsqueeze(1).expand(xyz_fine_sampled.shape)[:,:,-2:-1]
        rgb_fine, depth_fine, weights_fine = \
            rendering(model_MLP_dir, sigama_fine, feature_fine, rays_d,
                        dir_embedded, z_vals, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result



def render_sh(models,     
                embeddings,
                rays,
                world_size,
                grid_bounds,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                
                test_time=False,
                t_normalize = 0    #每个像素的时间相同
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    
    
    def rendering(sigama_coarse, feature_coarse, dir_,
                        dir_embedded, z_vals, weights_only=False):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        sh_deg = 2
        N_samples_ = feature_coarse.shape[1]   #[32768, 64, 3]
        # Embed directions
        feature_coarse = feature_coarse.view(-1, feature_coarse.shape[-1]) # (N_rays*N_samples_, 3)
        if not weights_only:   #[32768, 27]
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, 27)  
            dir_1 = torch.repeat_interleave(dir_, repeats=N_samples_, dim=0)
        # feature_dir = torch.cat([feature_coarse, dir_embedded], 1)
        # rgb = model_MLP_dir(feature_dir)
        # return rgb, rgb, rgb
        # Perform model inference to get rgb and raw sigma
        # B = feature_coarse.shape[0]   #32768 * 64   train 1024 * 64
        # out_chunks = []
        # for i in range(0, B, chunk):   #64次  每次像素点都选上了 每次是一个深度，重复64次
        #     # Embed positions by chunk
            
        #     feature_dir_ = feature_dir[i:i+chunk]
        #     out_chunks += [model_MLP_dir(feature_dir_)]   #每个list [1024*32,4] 64个list
        rgb = eval_sh(sh_deg, feature_coarse.reshape(
                *feature_coarse.shape[:-1],-1, (sh_deg + 1) ** 2), dir_1)
        # rgb = torch.cat(out_chunks, 0) #[2097152, 4]
        sigama = sigama_coarse.view(N_rays, N_samples_)
        rgbs= rgb.view(N_rays, N_samples_, 3)
        
        #给rgb增加一个激活函数sigmoid
        m = nn.Sigmoid()
        rgbs = m(rgbs)
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigama.shape, device=sigama.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigama+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights


    # Extract models from lists
    model_HashSiren = models[0]
    
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    xyz_min = grid_bounds[0].squeeze()
    xyz_max = grid_bounds[1].squeeze()
    
    # Decompose the inputs
    N_rays = rays.shape[0]   #32768
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    
    # #根据射线和AABB包围盒的交点重新得到near 和 far
    # t1 = []
    # t2 = []
    # for i in range(0,3):
        
    #     invdir = 1.0 / rays_d[:,i]
    #     t1 += [(xyz_min[i] - rays_o[:,i]) * invdir]
    #     t2 += [(xyz_max[i] - rays_o[:,i]) * invdir]
    # t1 = torch.stack(t1, dim = -1)
    # t2 = torch.stack(t2, dim = -1)
    # t = torch.cat((t1,t2),dim = -1)
    # a,_ = torch.sort(t, dim = -1)
    # distance = a[:,2:4]
    # rays_tmin = distance[:,0]
    # rays_tmax = distance[:,1]
    
    
    
    
    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, 27)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)   64个depth
    
    # #自己加的
    # z_steps = z_steps.expand(N_rays, N_samples)  #[32768, 64]   32768个像素点
    # z_vals = rays_tmin.unsqueeze(-1) * (1-z_steps) + rays_tmax.unsqueeze(-1) * z_steps

    
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)


    z_vals = z_vals.expand(N_rays, N_samples)  #[32768, 64]   32768个像素点
    ################
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)  [32768, 64, 3]
                         
                        
    # xyz_coarse_norm = ((xyz_coarse_sampled - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1
    xyz_coarse_norm = (((xyz_coarse_sampled - xyz_min) / (xyz_max - xyz_min)) * 2 - 1).float()
    output_feature = model_HashSiren(rays_o)
    
    
    output_feature = output_feature.reshape(world_size[2], world_size[1], world_size[0], 28).permute(3,0,1,2).float()
    sigama = output_feature[0:1]
    # #创建网格数据
    # import numpy as np
    # x = np.linspace(0, world_size[0]-1, world_size[0])
    # y = np.linspace(0, world_size[1]-1, world_size[1])
    # z = np.linspace(0, world_size[2]-1, world_size[2])
    # X, Y, Z = np.meshgrid(x, y, z)
    # coors = np.concatenate((X[:, :, :, None], Y[:, :, :, None], Z[:, :, :, None]), axis=-1)
    # sigama1 = sigama.permute(3,2,1,0).cpu().numpy()
    # coor = np.concatenate((coors,sigama1), axis=-1)

    # coor = coor.reshape(-1,4)
    # np.savetxt('/data1/liufengyi/all_datasets/second.txt',coor)
    # np.savetxt('/data/zhangruiqi/lfy/data/1.txt',coor)
    # from mayavi import mlab
    # import numpy as np
    # import pdb
    # def viz_mayavi(points):
    #     x = points[:, 2]  # x position of point
    #     y = points[:, 1]  # y position of point
    #     z = points[:, 0]  # z position of point
    #     pdb.set_trace()
    #     fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360)) #指定图片背景和尺寸
    #     mlab.points3d(x, y, z,
    #                         points[:, 3],          # Values used for Color，指定颜色变化依据
    #                         mode="point",
    #                         colormap='spectral', # 'bone', 'copper', 'gnuplot'


    #                         # color=(0, 1, 0),   # 也可以使用固定的RGB值


    #                         )


    #     mlab.show()
    import numpy as np
    x = np.linspace(0, world_size[0]-1, world_size[0])
    y = np.linspace(0, world_size[1]-1, world_size[1])
    z = np.linspace(0, world_size[2]-1, world_size[2])
    xx = np.repeat(x[None, :], len(y), axis=0)  # 第一个None对应Z，第二个None对应Y；所以后面是(len(z), len(y))
    xxx = np.repeat(xx[None, :, :], len(z), axis=0)
    yy = np.repeat(y[:, None], len(x), axis=1)
    yyy = np.repeat(yy[None, :, :], len(z), axis=0)
    zz = np.repeat(z[:, None], len(y), axis=1)
    zzz = np.repeat(zz[:, :, None], len(x), axis=2)
    coors = np.concatenate((zzz[:, :, :, None], yyy[:, :, :, None], xxx[:, :, :, None]), axis=-1)
    sigama1 = sigama.permute(1,2,3,0).cpu().numpy()
    coor = np.concatenate((coors,sigama1), axis=-1)

    coor = coor.reshape(-1,4)
    np.savetxt('/data/zhangruiqi/lfy/data/1_1.txt',coor)
    # # points = np.loadtxt("/data1/liufengyi/all_datasets/airplane_0001.txt", delimiter=',')

    # pdb.set_trace()
    # viz_mayavi(coor)
    # pdb.set_trace()
    
    
    
    feature_ = output_feature[1:]
    
    #TV loss
    tv_sigma = sigama.unsqueeze(0)
    tv_feature_ = feature_.unsqueeze(0)
    
    
    sigama_coarse = F.grid_sample(sigama.unsqueeze(0), xyz_coarse_norm.view(1,1,*xyz_coarse_norm.shape), mode='bilinear', align_corners=True).squeeze().unsqueeze(-1)
    feature_coarse = F.grid_sample(feature_.unsqueeze(0), xyz_coarse_norm.view(1,1,*xyz_coarse_norm.shape), mode='bilinear', align_corners=True).squeeze().permute(1,2,0)
    
    # if test_time:
    #     weights_coarse = \
    #         rendering(sigama_coarse, feature_coarse, rays_d,
    #                     dir_embedded, z_vals, weights_only=True)
    #     result = {'opacity_coarse': weights_coarse.sum(1)}
    # else:
    #     # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    #     rgb_coarse, depth_coarse, weights_coarse = \
    #         rendering(sigama_coarse, feature_coarse, rays_d,
    #                     dir_embedded, z_vals, weights_only=False)
    if test_time:
        weights_coarse = \
            rendering(sigama_coarse, feature_coarse, rays_d,
                        dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
        rgb_coarse, depth_coarse, weights_coarse = \
            rendering(sigama_coarse, feature_coarse, rays_d,
                        dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                    'depth_coarse': depth_coarse,
                    'opacity_coarse': weights_coarse.sum(1),
                    'tv_sigma' : tv_sigma,
                    'tv_feature_' : tv_feature_
                    }
    
    
    
    # t_normalize = t_normalize.unsqueeze(2).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    # if test_time:
    #     weights_coarse = \
    #         inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
    #                   dir_embedded, z_vals, weights_only=True, t_normalize = t_normalize)
    #     result = {'opacity_coarse': weights_coarse.sum(1)}
    # else:
    #     # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    #     rgb_coarse, depth_coarse, weights_coarse = \
    #         inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
    #                   dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize)
    #     result = {'rgb_coarse': rgb_coarse,
    #               'depth_coarse': depth_coarse,
    #               'opacity_coarse': weights_coarse.sum(1)
    #              }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        xyz_fine_norm = ((xyz_fine_sampled - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1
        
        sigama_fine = F.grid_sample(sigama.unsqueeze(0), xyz_fine_norm.view(1,1,*xyz_fine_norm.shape), mode='bilinear', align_corners=True).squeeze().unsqueeze(-1)
        feature_fine = F.grid_sample(feature_.unsqueeze(0), xyz_fine_norm.view(1,1,*xyz_fine_norm.shape), mode='bilinear', align_corners=True).squeeze().permute(1,2,0)
        # t_normalize_fine = t_normalize.unsqueeze(1).expand(xyz_fine_sampled.shape)[:,:,-2:-1]
        rgb_fine, depth_fine, weights_fine = \
            rendering(model_MLP_dir, sigama_fine, feature_fine, rays_d,
                        dir_embedded, z_vals, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result




def render_sh_sample(return_depth,
                models,
                embeddings,
                rays,
                world_size,
                grid_bounds,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                
                test_time=False,
                t_normalize = 0    #每个像素的时间相同
                ):
    """
    把box缩小,
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    
    
    def rendering(sigama_coarse, feature_coarse, dir_,
                        dir_embedded, z_vals, weights_only=False,):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        sh_deg = 2
        N_rays = feature_coarse.shape[0]
        N_samples_ = feature_coarse.shape[1]   #[32768, 64, 3]
        # Embed directions
        feature_coarse = feature_coarse.view(-1, feature_coarse.shape[-1]) # (N_rays*N_samples_, 3)
        if not weights_only:   #[32768, 27]
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, 27)  
            dir_1 = torch.repeat_interleave(dir_, repeats=N_samples_, dim=0)
        # feature_dir = torch.cat([feature_coarse, dir_embedded], 1)
        # rgb = model_MLP_dir(feature_dir)
        # return rgb, rgb, rgb
        # Perform model inference to get rgb and raw sigma
        # B = feature_coarse.shape[0]   #32768 * 64   train 1024 * 64
        # out_chunks = []
        # for i in range(0, B, chunk):   #64次  每次像素点都选上了 每次是一个深度，重复64次
        #     # Embed positions by chunk
            
        #     feature_dir_ = feature_dir[i:i+chunk]
        #     out_chunks += [model_MLP_dir(feature_dir_)]   #每个list [1024*32,4] 64个list
        rgb = eval_sh(sh_deg, feature_coarse.reshape(
                *feature_coarse.shape[:-1],-1, (sh_deg + 1) ** 2), dir_1)
        # rgb = torch.cat(out_chunks, 0) #[2097152, 4]
        sigama = sigama_coarse.view(N_rays, N_samples_)
        rgbs= rgb.view(N_rays, N_samples_, 3)
        
        #给rgb增加一个激活函数sigmoid
        m = nn.Sigmoid()
        rgbs = m(rgbs)
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigama.shape, device=sigama.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigama+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights


    # Extract models from lists
    # import time
    # torch.cuda.synchronize()
    # start = time.time()
    
    output_feature_grid = torch.zeros((world_size.prod(), 28), device = 'cuda')
    model_HashSiren = models[0]
    
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    xyz_min = grid_bounds[0].squeeze()
    xyz_max = grid_bounds[1].squeeze()
    
    # Decompose the inputs
    N_rays = rays.shape[0]   #32768
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    
    #根据射线和AABB包围盒的交点重新得到near 和 far
    t1 = []
    t2 = []
    for i in range(0,3):
        # i = 2
        invdir = 1.0 / rays_d[:,i]
        t1 += [(xyz_min[i] - rays_o[:,i]) * invdir]
        t2 += [(xyz_max[i] - rays_o[:,i]) * invdir]

    
    tMin = torch.stack(t1, dim = -1)   #最小
    tMax = torch.stack(t2, dim = -1)   #最大
    t1 = torch.min(tMin, tMax)
    t2 = torch.max(tMin, tMax)
    tNear = torch.max(torch.max(t1[:,0], t1[:,1]), t1[:,2])
    tFar = torch.min(torch.min(t2[:,0], t2[:,1]), t2[:,2])
    box_index = (tNear<tFar)
    if box_index.sum() == 0:
        rgb_coarse1 = torch.ones((N_rays, 3), device = 'cuda')
        depth_coarse1 = torch.zeros((N_rays, 1), device = 'cuda')
        result = {'rgb_coarse': rgb_coarse1,
                    'depth_coarse': depth_coarse1,}
        return result
    # tNear = tNear[box_index]
    # tFar = tFar[box_index]
    rays_tmin = tNear
    rays_tmax = tFar
    
    # t1 = torch.stack(t1, dim = -1)   #最小
    # t2 = torch.stack(t2, dim = -1)   #最大
    # t = torch.cat((t1,t2),dim = -1)
    # a,_ = torch.sort(t, dim = -1)
    # distance = a[:,2:4]
    # rays_tmin = distance[:,0]
    # rays_tmax = distance[:,1]
    # rays_tmin = rays_tmin.min()
    # rays_tmax = rays_tmax.max()
    # near = rays_tmin
    # far = rays_tmax
    
    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, 27)
    
    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)   64个depth
    
    #自己加的
    z_steps = z_steps.expand(N_rays, N_samples)  #[32768, 64]   32768个像素点
    z_vals = rays_tmin.unsqueeze(-1) * (1-z_steps) + rays_tmax.unsqueeze(-1) * z_steps

    
    # if not use_disp: # use linear sampling in depth space
    #     z_vals = near * (1-z_steps) + far * z_steps
    # else: # use linear sampling in disparity space
    #     z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)


    # z_vals = z_vals.expand(N_rays, N_samples)  #[32768, 64]   32768个像素点
    ################
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand
    rays_o = rays_o[box_index]
    rays_d = rays_d[box_index]
    z_vals = z_vals[box_index]
    xyz_coarse_sampled = (rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)) # (N_rays, N_samples, 3)  [32768, 64, 3]
                         
    
    # xyz_coarse_norm = ((xyz_coarse_sampled - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1
    xyz_coarse_norm = ((((xyz_coarse_sampled - xyz_min) / (xyz_max - xyz_min))) * 2 - 1).float()
    xyz_coarse_grid = ((xyz_coarse_norm+1)/2*((world_size.cuda()-1).unsqueeze(0).unsqueeze(1)).cuda())
    gird2MLP_left = torch.floor(xyz_coarse_grid)
    voxel_indices = gird2MLP_left.reshape(-1, 3).unsqueeze(1) + BOX_OFFSETS
    
    voxel_indices = torch.unique(voxel_indices.reshape(-1,3),dim=0)
    voxel_indices[:,0] = torch.clamp(voxel_indices[:,0],0,world_size[0]-1)
    voxel_indices[:,1] = torch.clamp(voxel_indices[:,1],0,world_size[1]-1)
    voxel_indices[:,2] = torch.clamp(voxel_indices[:,2],0,world_size[2]-1)
    voxel_indices = torch.unique(voxel_indices.reshape(-1,3),dim=0)
    # torch.cuda.synchronize()
    # start11 = time.time()  
    # print("others   :",start1-start)
    voxel_indices = (voxel_indices[:,0]*world_size[1]*world_size[2]+voxel_indices[:,1]*world_size[2]+voxel_indices[:,2]).long()
    # torch.cuda.synchronize()
    # start1 = time.time() 
    # print("model1   :",start1-start11)
    output_feature = model_HashSiren(voxel_indices, world_size)
    # output_feature_grid = torch.zeros(world_size.prod(), 28).cuda()
    
    # torch.cuda.synchronize()
    # start111 = time.time() 
    # print("model    :",start111-start1) 
    # output_feature_grid[voxel_indices[:,0].tolist(),voxel_indices[:,1].tolist(),voxel_indices[:,2].tolist(),:] = output_feature.float()
    output_feature_grid[voxel_indices] = output_feature.float()
    output_feature_grid = output_feature_grid.reshape(*world_size,-1)
    # torch.cuda.synchronize()
    # start2 = time.time()  
    # print("others   :",start2-start1)
    sigama = output_feature_grid.permute(3,2,1,0)[0:1]
    feature_ = output_feature_grid.permute(3,2,1,0)[1:]
    
    # torch.cuda.synchronize()
    # start2 = time.time()
    # print("model    :",start2-start1)
    
    # output_feature = output_feature.reshape(world_size[2], world_size[1], world_size[0], 28).permute(3,0,1,2).float()
    # sigama = output_feature[0:1]
    # #创建网格数据
    # import numpy as np
    # x = np.linspace(0, world_size[0]-1, world_size[0])
    # y = np.linspace(0, world_size[1]-1, world_size[1])
    # z = np.linspace(0, world_size[2]-1, world_size[2])
    # X, Y, Z = np.meshgrid(x, y, z)
    # coors = np.concatenate((X[:, :, :, None], Y[:, :, :, None], Z[:, :, :, None]), axis=-1)
    # sigama1 = sigama.permute(3,2,1,0).cpu().numpy()
    # coor = np.concatenate((coors,sigama1), axis=-1)

    # coor = coor.reshape(-1,4)
    # np.savetxt('/data1/liufengyi/all_datasets/second.txt',coor)
    # np.savetxt('/data/zhangruiqi/lfy/data/1.txt',coor)
    # from mayavi import mlab
    # import numpy as np
    # import pdb
    # def viz_mayavi(points):
    #     x = points[:, 2]  # x position of point
    #     y = points[:, 1]  # y position of point
    #     z = points[:, 0]  # z position of point
    #     pdb.set_trace()
    #     fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360)) #指定图片背景和尺寸
    #     mlab.points3d(x, y, z,
    #                         points[:, 3],          # Values used for Color，指定颜色变化依据
    #                         mode="point",
    #                         colormap='spectral', # 'bone', 'copper', 'gnuplot'


    #                         # color=(0, 1, 0),   # 也可以使用固定的RGB值


    #                         )


    #     mlab.show()


    # # points = np.loadtxt("/data1/liufengyi/all_datasets/airplane_0001.txt", delimiter=',')

    # pdb.set_trace()
    # viz_mayavi(coor)
    # pdb.set_trace()
    
    
    
    # feature_ = output_feature[1:]
    
    #TV loss
    # tv_sigma = sigama.unsqueeze(0)
    # tv_feature_ = feature_.unsqueeze(0)
    
    
    sigama_coarse = F.grid_sample(sigama.unsqueeze(0), xyz_coarse_norm.view(1,1,*xyz_coarse_norm.shape), mode='bilinear', align_corners=True).squeeze().unsqueeze(-1)
    feature_coarse = F.grid_sample(feature_.unsqueeze(0), xyz_coarse_norm.view(1,1,*xyz_coarse_norm.shape), mode='bilinear', align_corners=True).squeeze().permute(1,2,0)
    # torch.cuda.synchronize()
    # start3 = time.time()
    # print("sampling :",start3-start2)
    # if test_time:
    #     weights_coarse = \
    #         rendering(sigama_coarse, feature_coarse, rays_d,
    #                     dir_embedded, z_vals, weights_only=True)
    #     result = {'opacity_coarse': weights_coarse.sum(1)}
    # else:
    #     # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    #     rgb_coarse, depth_coarse, weights_coarse = \
    #         rendering(sigama_coarse, feature_coarse, rays_d,
    #                     dir_embedded, z_vals, weights_only=False)
    # torch.cuda.synchronize()
    # start4 = time.time() 
    if test_time:
        weights_coarse = \
            rendering(sigama_coarse, feature_coarse, rays_d,
                        dir_embedded, z_vals, weights_only=True,)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
        rgb_coarse, depth_coarse, weights_coarse = \
            rendering(sigama_coarse, feature_coarse, rays_d,
                        dir_embedded, z_vals, weights_only=False,)
        rgb_coarse1 = torch.ones((N_rays, 3), device = 'cuda')
        rgb_coarse1[box_index] = rgb_coarse
        # rgb_coarse[~box_index] = 1
        # end = time.time() 
        # print("rendering:",end-start4)
        if return_depth:
            depth_coarse1 = torch.zeros((N_rays, 1), device = 'cuda')
            depth_coarse1[box_index] = depth_coarse.unsqueeze(-1)
            result = {'rgb_coarse': rgb_coarse1,
                    'depth_coarse': depth_coarse1,}
        else:
            result = {'rgb_coarse': rgb_coarse1,
                #   'box_index': box_index,
                    # 'depth_coarse': depth_coarse,
                    # 'opacity_coarse': weights_coarse.sum(1),
                    # 'tv_sigma' : tv_sigma,
                    # 'tv_feature_' : tv_feature_
                    }
    
    
    
    # t_normalize = t_normalize.unsqueeze(2).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    # if test_time:
    #     weights_coarse = \
    #         inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
    #                   dir_embedded, z_vals, weights_only=True, t_normalize = t_normalize)
    #     result = {'opacity_coarse': weights_coarse.sum(1)}
    # else:
    #     # t_normalize_coarse = t_normalize.unsqueeze(1).expand(xyz_coarse_sampled.shape)[:,:,-2:-1]
    #     rgb_coarse, depth_coarse, weights_coarse = \
    #         inference(model_coarse, model_latent, embedding_xyz, xyz_coarse_sampled, rays_d,
    #                   dir_embedded, z_vals, weights_only=False, t_normalize = t_normalize)
    #     result = {'rgb_coarse': rgb_coarse,
    #               'depth_coarse': depth_coarse,
    #               'opacity_coarse': weights_coarse.sum(1)
    #              }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        xyz_fine_norm = ((xyz_fine_sampled - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1
        
        sigama_fine = F.grid_sample(sigama.unsqueeze(0), xyz_fine_norm.view(1,1,*xyz_fine_norm.shape), mode='bilinear', align_corners=True).squeeze().unsqueeze(-1)
        feature_fine = F.grid_sample(feature_.unsqueeze(0), xyz_fine_norm.view(1,1,*xyz_fine_norm.shape), mode='bilinear', align_corners=True).squeeze().permute(1,2,0)
        # t_normalize_fine = t_normalize.unsqueeze(1).expand(xyz_fine_sampled.shape)[:,:,-2:-1]
        rgb_fine, depth_fine, weights_fine = \
            rendering(model_MLP_dir, sigama_fine, feature_fine, rays_d,
                        dir_embedded, z_vals, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result
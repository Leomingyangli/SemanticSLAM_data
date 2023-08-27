import sys
# import pdb
import copy
import math
import numpy as np

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
# from timeit import default_timer as timer

import scipy.integrate as it
# from mapnet.utils import norm_angle

# torch.set_printoptions(precision=4,linewidth=100,sci_mode=False)
def rotate_tensor(r, t, mode='bilinear'):
    """
    Inputs:
        r     - (bs, f, h, w) Tensor
        t     - (bs, ) Tensor of angles
    Outputs:
        r_rot - (bs, f, h, w) Tensor
    """
    device = r.device

    sin_t  = torch.sin(t)
    cos_t  = torch.cos(t)

    # This R convention means Y axis is downwards.
    A      = torch.zeros(r.size(0), 2, 3).to(device)

    # rotate counter clockwise
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t

    grid   = F.affine_grid(A, r.size(), align_corners=False)
    r_rot  = F.grid_sample(r, grid,mode=mode, align_corners=False)

    return r_rot

def rotation_resample(x, angles, mode='bilinear', offset=0.0):
    """
    Inputs:
        x       - (bs, f, s, s) feature maps
        angles_ - (nangles, ) set of angles to sample
    Outputs:
        x_rot   - (bs, nangles, f, s, s)
    """
    # angles      = angles_.clone() # (nangles, )
    bs          = x.shape[0]
    nangles     = angles.shape[0]

    x_rep       = x.unsqueeze(1).expand(-1, nangles, -1, -1, -1) # (bs, nangles, f, s, s)
    x_rep       = rearrange(x_rep, 'b o e h w -> (b o) e h w')
    angles      = angles.unsqueeze(0).expand(bs, -1).contiguous().view(-1) # (bs * nangles, )
    x_rot       = rotate_tensor(x_rep, angles + math.radians(offset), mode) # (bs * nangles, f, s, s)
    x_rot       = rearrange(x_rot, '(b o) e h w -> b o e h w', b=bs)

    return x_rot

def localize(x_rot, maps):
    """ 
    Localize input features in maps
    Inputs: 
        x_rot  - (bs, nangles, f, s, s) ground projection of image 
        maps   - (bs, f, H, W) full map 
        angles - (nangles, ) set of angles to sample
    Outputs:
        poses  - (bs, nangles, H, W) softmax over all poses
        x_rot  - (bs, nangles, f, s, s)
    """ 
    bs, nangles, f, s, s = x_rot.shape
    _, _, H, W = maps.shape

    # Look at https://github.com/BarclayII/hart-pytorch/blob/master/dfn.py 
    # for how batch-wise convolutions are performed
    x_rot = rearrange(x_rot, 'b o e h w -> (b o) e h w')
    maps  = rearrange(maps, 'b e h w -> () (b e) h w')
    # print('localize: ', x_rot.dtype, maps.dtype)
    poses = F.conv2d(maps, x_rot, stride=1, padding=s//2, groups=bs) # (1, bs*nangles, H, W)
    poses = F.softmax(rearrange(poses, '() (b o) h w -> b (o h w)', b=bs), dim=1)
    poses = rearrange(poses, 'b (o h w) -> b o h w', h=H, w=W)

    return poses


def imu_cross_check(v, i, p_v, mask_model, threshold):
    '''
        Leverage imu to reduce uncertainty from visual estimation
            v: allocentric visual pose (b,3)
            i: allocentric inertial pose (b,3)
            p_v: maximal probability of visual probability distribution (b,3)
            mask_model: the model to determine which mask procedure are used
                        'shift' - threshold = (pose,deg)
                        'confidence' - threshold = scalar
    '''
    bs = v.shape[0]
    if mask_model == 'shift':
        diff_pos = torch.linalg.norm(v[:,:2] - i[:,:2],ord=2,dim=1) #b
        diff_ori = torch.linalg.norm(v[:,2] - i[:,2]) #b
        choice_v2i = ((diff_pos > threshold[0]) & (diff_ori > threshold[1])).long()
    elif mask_model == 'confidence':
        choice_v2i = (p_v >= threshold).long()

    pose = torch.stack([v,i],dim=1)[range(bs),choice_v2i] #b,2,3->b,3    

    return pose, choice_v2i

def mask_generated(pose, mask_shape, mask_dia):
    '''
    Generate binary mask for each position of grid cell
    input:
        p: (b,3) pose
        mask_shape: (bs,h,w)
        mask_dia: odd int, the window size

    output:
        mask: (b,h,w)
    '''
    if mask_dia < 2: return 1

    #generate mask
    device = pose.device
    bs, h, w = mask_shape
    mask = torch.zeros(mask_shape,device=device) #bs,h,w
    rnge_row = rnge_col = torch.arange(-(mask_dia//2),mask_dia//2+1,device=device).expand(bs, mask_dia) #(bs,mask_dia)
    # generate index for different center and window size
    row = torch.clamp(pose[:,[0]] + rnge_row, 0,h-1).long() #(bs,mask_dia)
    col = torch.clamp(pose[:,[1]] + rnge_col, 0,w-1).long()
    mask[torch.arange(bs).long()[:,None,None], row[:,:,None], col[:,None,:]] = 1

    return mask

def project_to_ground_plane(img_feats, spatial_locs, valid_inputs, local_shape, K, eps=-1e16):
    """
    Project image features to locations in ground-plane given by spatial_locs.
    Inputs:
        img_feats       - (bs, f, H/K, W/K) image features to project to ground plane
        spatial_locs    - (bs, 2, H, W)
                          for each pixel in each batch, the (x, y) ground-plane locations are given.
        valid_inputs    - (bs, 1, H, W) ByteTensor
        local_shape     - (outh, outw) tuple indicating size of output projection
        K               - image_size / map_shape ratio (needed for sampling values from spatial_locs)
        eps             - fill_value
    Outputs:
        proj_feats      - (bs, f, s, s)
    """
    device = img_feats.device
    outh, outw = local_shape
    bs, f, HbyK, WbyK = img_feats.shape
    # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
    idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(device), \
                (torch.arange(0, WbyK, 1)*K).long().to(device))
    # print('idxes_ss:', idxes_ss)
    spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
    valid_inputs_ss = valid_inputs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 1, HbyK, WbyK)
    valid_inputs_ss = valid_inputs_ss.squeeze(1) # (bs, HbyK, WbyK)
    invalid_inputs_ss = ~valid_inputs_ss

    # print('spatial_locs:', spatial_locs.shape , spatial_locs[0,0,0:10,:])
    # print('spatial_locs_ss:', spatial_locs_ss.shape , spatial_locs_ss[0,0,0:10,:])


    # Filter out invalid spatial locations
    invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                           (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)

    invalid_writes = invalid_spatial_locs | invalid_inputs_ss
    # Set the idxes for all invalid locations to (0, 0)
    spatial_locs_ss[:, 0][invalid_writes] = 0
    spatial_locs_ss[:, 1][invalid_writes] = 0
    # print('spatial_locs_ss_final:', spatial_locs_ss.shape , spatial_locs_ss[0,0,:,:])
    # print('valid_inputs:', valid_inputs.shape , valid_inputs[0,0,0:10,:])
    # print('ivalid_inputs_ss:', invalid_inputs_ss.shape , invalid_inputs_ss[0,:,:])
    # print('invalid_writes:', invalid_writes.shape , invalid_writes[0])
    # Weird hack to account for max-pooling negative feature values
    invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').float()
    img_feats_masked = img_feats * (1 - invalid_writes_f) + eps * invalid_writes_f # add -1e16 to invalid place
    # print('img_feats:', img_feats.shape, img_feats[0,0])
    # print('img_feats_masked:', img_feats_masked.shape, img_feats_masked[0,0])
    img_feats_masked = rearrange(img_feats_masked, 'b e h w -> b e (h w)')
    # Linearize ground-plane indices (linear idx = y * W + x)
    linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
    linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
    linear_locs_ss = linear_locs_ss.expand(-1, f, -1) # .contiguous()
    # print('linear_locs_ss:', linear_locs_ss.shape, linear_locs_ss[0,0])
    proj_feats, _ = torch_scatter.scatter_max(
                        img_feats_masked,
                        linear_locs_ss,
                        dim=2,
                        dim_size=outh*outw,
                        # fill_value=eps,
                    )
    proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)
    # print('proj_feats:', proj_feats.shape, proj_feats[0,0])
    # Replace invalid features with zeros
    eps_mask = (proj_feats <= eps)
    proj_feats[eps_mask] = 0
    # print('proj_feats_final:', proj_feats.shape, proj_feats[0,0])
    # print('eps_mask:', eps_mask.shape, eps_mask[0,0])
    # print('spatial_locs(bs, 2, H, W):',spatial_locs.shape,spatial_locs[0,:,0:9,0:9],sep='\n')
    # print('img_feature(bs,f,H/K,w/K):',img_feats.shape,img_feats[0,0,0:9,0:9],sep='\n')
    # print('valid_inputs(bs, 1, H, W):',valid_inputs.shape,valid_inputs[0,:,0:9,0:9],sep='\n')
    # print('proj_feats(bs, f, s, s):',proj_feats.shape,proj_feats[0,0],sep='\n')
    return proj_feats

def compute_spatial_locs(depth_inputs, local_shape, local_scale):
    """
    Compute locations on the ground-plane for each pixel.
    Inputs:
        depth_inputs  - (bs, 3, imh, imw) depth values per pixel in `units`. 
        local_shape   - (s, s) tuple of ground projection size 51
        local_scale   - cell size of ground projection in `units` 0.1m
    Outputs:
        spatial_locs  - (bs, 2, imh, imw) x,y locations of projection per pixel
        valid_inputs  - (bs, 1, imh, imw) ByteTensor (all locations where depth measurements are available)
    """
    s               = local_shape[1]
    X               = depth_inputs[:,[0],:,:]
    Z               = depth_inputs[:,[2],:,:]
    valid_inputs    = ~torch.isnan(depth_inputs[:,[0],:,:])

    # Note: map_scale - dimension of each grid in meters
    # camera(depth) map, X go rightward, and Z go upward
    # local map, X go rightward, Y go downward
    x_gp = ( (X / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)
    y_gp = (-(Z / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)
    return torch.cat([x_gp, y_gp], dim=1), valid_inputs

def add_imu_to_pose(imu, pre_pose, freq):
    '''
    Calculate pose candidates based on IMU data and Last pose
    input:
        imu     : (t, B, 3)
                    ax,ay,w_yaw -- linear accelaration for x,y; angular velocity for yaw
        pre_pose: (B,3) predicted pose from last step
        freq    : (1) sampling rate of imu data

    Output:
        absolute pose (t,B,3)
    '''
    axy     = imu[:,:,:2] #(t,B,2)
    w_yaw   = imu[:,:,2] #(t,B)
    device = pre_pose.device
    vxy             = it.cumtrapz(axy, dx=1/freq, axis=0, initial=0)
    displacement    = torch.tensor(it.cumtrapz(vxy, dx=1/freq, axis=0, initial=0))[-1,...].to(device) #B,2
    yaw             = torch.tensor(it.cumtrapz(w_yaw, dx=1/freq, axis=0, initial=0))[-1,:,None].to(device) #B,1
    rela_pose       = torch.cat(displacement,yaw, 1)
    abs_pose = pre_pose + rela_pose

    return abs_pose

def crop_map(maps, pred_poses, abs_poses, scale, local_map_size):
    """
    Get a small piece of maps given poses and size
    input:
            maps            : (bs, f, map_size, map_size)  
                                X rightward, Y downward, origin at top-left
            pred_poses      : (bs, 2or3) x, y, yaw = imu_pose/predicted poses
                               x&y range = [0,local_map_size]
            abs_poses      : (bs, 2) x, y, yaw = imu_pose/predicted poses
                               x&y range = [0,map_size]
            local_map_size  : scalar
    output:
            partial_map     : (bs, f, local_map_size, local_map_size)
            offset          : (bs, 2) (x,y)
    """

    bs, f, map_size, _ = maps.shape
    device = maps.device
    # edge length / center index of partial center
    edge = torch.tensor((local_map_size-1)/2).long()
    #find center index of the maps
    ori = torch.tensor((map_size-1)/2)
    #find  pose on global map-> pori_pmap - pori_gmap = pose_pmap - pose_gmap
    index_poses = torch.clamp(pred_poses[:,:2] - edge + abs_poses[:,:2],min=0,max=map_size).long()
    # index_poses = (torch.round(index_poses/scale)).long()
    # index_poses = torch.round(index_poses).long()
    partial_maps = torch.full((bs, f, local_map_size, local_map_size),0,dtype=torch.float32).to(device)

    # print("pred_poses: ",pred_poses)
    # print("index_poses: ", index_poses)
    #Crop the map for each batch
    for b in range(bs):
        # print('ini l,r,u,d',index_poses[b,0]-edge,index_poses[b,0]+edge+1,index_poses[b,1]-edge,index_poses[b,1]+edge+1)
        # consider the boundary
        up = max(0, index_poses[b,1]-edge)
        down = min(map_size, index_poses[b,1]+edge+1)
        left = max(0, index_poses[b,0]-edge)
        right = min(map_size, index_poses[b,0]+edge+1)
        one_map = maps[b, :, up:down, left:right]
        # print('bef l,r,u,d',left,right,up,down)
        # padding boundary with 0
        if up == 0:
            up -= index_poses[b,1]-edge
            down -= index_poses[b,1]-edge
        elif down == map_size:
            down -= up
            up = torch.tensor(0)
        else:
            up = torch.tensor(0)
            down = local_map_size
        if left == 0:
            left -= index_poses[b,0]-edge
            right -= index_poses[b,0]-edge
        elif right == map_size:
            right -= left
            left = torch.tensor(0)
        else:
            left = torch.tensor(0)
            right = local_map_size
        # print('aft l,r,u,d',left,right,up,down)
        partial_maps[b, :, up:down, left:right] = one_map
        # print('partial_maps','\n',partial_maps.dtype)
    
    return partial_maps, index_poses.int()

def enlarge_obs(o_reg, maps, abs_pose):
    """
    Inputs:
        o_reg   - (bs, f, h, w) Tensor of ground plane projection of current image on partial map
        maps     - (bs, f, H, W) Tensor of overall map
        abs_pose  - (bs, 2) torch.long pose in the global map
    Outputs:
        large_o_reg  - (bs, f, H, W) Tensor of ground plane projection of current image on overall map
    """
    _, _, H, W = maps.shape
    bs, f, h, w = o_reg.shape
    large_o_reg = torch.zeros((bs,f,H,W), device=maps.device)
    # ori_H = torch.tensor((H-1)/2, device=maps.device).long()
    # ori_W = torch.tensor((W-1)/2, device=maps.device).long()
    e_h = torch.tensor((h-1)/2, device=maps.device).long()
    e_w = torch.tensor((w-1)/2, device=maps.device).long()
    # print(ori_H, ori_W, e_h, e_w)
    for b in range(bs):
        # consider the boundary
        up =    torch.clamp(abs_pose[b,1]-e_h,  min=0, max=H).int()
        down =  torch.clamp(abs_pose[b,1]+e_h+1,min=0, max=H).int()
        left =  torch.clamp(abs_pose[b,0]-e_w,  min=0, max=W).int()
        right = torch.clamp(abs_pose[b,0]+e_w+1,min=0, max=W).int()
        if up==0:
            s_up = -(abs_pose[b,1]-e_h)
            s_down = down - (abs_pose[b,1]-e_h)
        elif down==H:
            s_down = down - up
            s_up = torch.tensor(0)
        else:
            s_up = torch.tensor(0)
            s_down = h
        if left==0:
            s_left = -(abs_pose[b,0]-e_w)
            s_right = right - (abs_pose[b,0]-e_w)
        elif right==W:
            s_right = right - left
            s_left = torch.tensor(0)
        else:
            s_left = torch.tensor(0)
            s_right = w
        # print(up,down, left,right,s_up,s_down,s_left,s_right)
        large_o_reg[b, :,
                    up:down,
                    left:right] = o_reg[b,:,s_up:s_down,s_left:s_right]
    return large_o_reg

def normalize_data(pre_data):
    '''Normalaize the data:
    input: Bx3 or Bxtx3 , pytorch tensor
    output: normalized data with same dimensions
    '''
    minimum = torch.min(pre_data, dim=-2)
    maximum = torch.max(pre_data, dim=-2)
    aft_data = (pre_data - minimum) / (maximum - minimum)
    return aft_data

class OUActionNoise(object):
    def __init__(self, mu, sigma=1, theta=0.2, dt=0.1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(scale=1, size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)

def get_new_pose_batch(pose, rel_pose_change):
    pose[:, 1] += rel_pose_change[:, 0] * torch.sin(pose[:, 2]) + rel_pose_change[:, 1] * torch.cos(pose[:, 2])
    pose[:, 0] += rel_pose_change[:, 0] * torch.cos(pose[:, 2]) - rel_pose_change[:, 1] * torch.sin(pose[:, 2])
    pose[:, 2] += rel_pose_change[:, 2]

    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0
    return pose

def unravel_index(
    indices: torch.LongTensor,
    shape,
    ) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """
    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')
    coord = torch.stack(coord[::-1], dim=-1)

    return coord

def gnrt_mask(mask_shape,center,mask_offset,device):
    '''
    Product mask for different center and window size of batches
    The key is to generate index, inplementing batch-wise slicing with different start and stop index without loop

    input:
        mask_shape: tuple (bs,h,w), shape of mask, same as tensor to be masked(except feature channel)
        center: (bs,3) or (bs,2) pose of agent
        mask_offset: (bs) size of mask window
    output:
        mask: (bs,h,w) 1/0 tensor
            1 to keep and 0 to be masked
    '''
    bs,h,w = mask_shape
    mask = torch.zeros(mask_shape,device=device) #(bs,h,w)

    # generate range given offset for each batch
    max_len = mask_offset.max().int() #bs
    min_ = -mask_offset[:,None].expand(bs,2*max_len+1).int() #(bs,2*max_len+1)
    max_ = mask_offset[:,None].expand(bs,2*max_len+1).int()
    rnge_row = rnge_col = torch.clamp(torch.arange(-max_len,max_len+1,device=device).expand(bs, 2*max_len+1),min=min_,max=max_) #(bs,2*max_len+1)
    
    # generate index for different center and window size
    row = torch.clamp(center[:,[0]] + rnge_row, 0,h-1).long() #(bs,2*max_len+1)
    col = torch.clamp(center[:,[1]] + rnge_col, 0,w-1).long()

    mask[torch.arange(bs).long()[:,None,None], row[:,:,None], col[:,None,:]] = 1 
    return mask

def allocentric_poses(ego_pose, prev_poses):
    '''
        input: [bs,3] x, y, yaw
            where each pose is relative its last pose

        output:[bs,3]
            where each pose is relative to start pose
    '''
    poses = prev_poses.clone()
    poses[:,0] = prev_poses[:,0] + ego_pose[:,0] * torch.cos(prev_poses[:,2]) - ego_pose[:,1] * torch.sin(prev_poses[:,2])
    poses[:,1] = prev_poses[:,1] + ego_pose[:,0] * torch.sin(prev_poses[:,2]) + ego_pose[:,1] * torch.cos(prev_poses[:,2])
    poses[:,2] = norm_angle(prev_poses[:,2] + ego_pose[:,2])
    return poses

def allocentric_poses_all(ego_poses):
    '''
        input: [L,3] x, y, yaw
            where each pose is relative its last pose

        output:[L,3]
            where each pose is relative to start pose
    '''
    # Initialize the allocentric pose as the first pose in the sequence
    allo_pose = list(ego_poses[0].numpy())
    # Initialize an empty list to store the output poses
    output = [allo_pose]
    # Loop through the rest of the poses in the sequence
    for ego_pose in ego_poses[1:]:
        # Calculate the allocentric position and orientation of the current pose
        allo_x = allo_pose[0] + ego_pose[0] * math.cos(allo_pose[2]) - ego_pose[1] * math.sin(allo_pose[2])
        allo_y = allo_pose[1] + ego_pose[0] * math.sin(allo_pose[2]) + ego_pose[1] * math.cos(allo_pose[2])
        allo_theta = (allo_pose[2] + ego_pose[2]) % (2 * math.pi)
        if allo_theta > math.pi:
            allo_theta -= 2 * math.pi
        elif allo_theta < -math.pi:
            allo_theta += 2 * math.pi
        # Update the allocentric pose with the current pose
        allo_pose = [allo_x, allo_y, allo_theta]
        # Append the allocentric pose to the output list
        output.append(allo_pose)
    # Return the output list of relative poses
    output = torch.tensor(output, dtype=ego_poses.dtype ,device=ego_poses.device)
    
    return output
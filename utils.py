import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
from einops import rearrange, reduce, asnumpy
from scipy.ndimage import rotate
'''
Maintainance History:
utils
----------------------------
ver 1.0 - Nov 23th 2020
    Chnage parts of following function
    - convert_world2map()
    - convert_map2world()
    - compute_relative_pose()
ver 2.0 
    - process_image()
'''
def flatten_two(x):
    try:
        # print("faltten",x.shape)
        return x.view(-1, *x.shape[2:])
    except:
        print("exp faltten",x.shape)
        return x.contiguous().view(-1, *x.shape[2:])

def unflatten_two(x, sh1, sh2):
    try: 
        # print("unfaltten",x.shape,sh1,sh2)
        return x.view(sh1, sh2, *x.shape[1:])
    except:
        print("exp unfaltten",x.shape,sh1,sh2)
        return x.contiguous().view(sh1, sh2, *x.shape[1:])

def get_camera_parameters(env_name, obs_shape):
    # Note: obs_shape[*]/K done because image features and depth will be downsampled by K
    # These values are obtained from Matterport3D files. 
    if env_name == 'avd':
        K   = 1 # orig image size / feature map size
        fx  = 1070.00 * (obs_shape[2]) / 1920.0
        fy  = 1069.12 * (obs_shape[1]) / 1080.0
        cx  = 927.269 * (obs_shape[2]) / 1920.0
        cy  = 545.760 * (obs_shape[1]) / 1080.0
        fov = math.radians(75)
    else:
        K   = None 
        fx  = None 
        fy  = None 
        cx  = None 
        cy  = None 
        fov = None 

    camera_parameters = {'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'fov': fov}
    return camera_parameters

# Normalize angles between -pi to pi
def norm_angle(x):
    # x - torch Tensor of angles in radians
    return torch.atan2(torch.sin(x), torch.cos(x))

def process_image(img,mode='train'):
    # img - (bs, C, H, W)
    # mean = [0.782, 0.779, 0.776]
    # std  = [0.307, 0.309, 0.311]
    if mode == 'train':
        mean = [0.783, 0.780, 0.777]
        std  = [0.307, 0.308, 0.311]
    else:
        mean = [0.792, 0.789, 0.785]
        std  = [0.302, 0.304, 0.308]
    img_proc = img.float() / 255.0

    img_proc[:, 0] = (img_proc[:, 0] - mean[0]) / std[0]
    img_proc[:, 1] = (img_proc[:, 1] - mean[1]) / std[1]
    img_proc[:, 2] = (img_proc[:, 2] - mean[2]) / std[2]

    return img_proc

def convert_world2map_ori(pose, map_shape, map_scale, angles):
    """
    Inputs:
        pose - (bs, 3) Tensor agent pose in world coordinates (x, y, theta) [0,10] [0,10] [-pi,pi]
        map_shape - (f, h, w) tuple     32x31x31
        map_scale - scalar              1
        angles - (nangles, ) Tensor     [-pi,pi) dim=12
        eps - scalar angular bin-size   

    Conventions:
        world positions - X Upward, Y rightward, origin at center
        map positions - X rightward, Y downward, origin at top-left
    """
    x = pose[:, 0]
    y = pose[:, 1]
    mh, mw = map_shape[1:]
    nangles = float(angles.shape[0])
    # This convention comes from transform_to_map() in model_pose.py [0,100] when mw=101 
    ref_on_map_x = torch.clamp((mw-1)/2 + x/map_scale, 0, mw-1).round().long() 
    ref_on_map_y = torch.clamp((mh-1)/2 - y/map_scale, 0, mh-1).round().long()
    # Mapping heading angles to map locations [6,7,8,9,10,11,0,1,2,3,4,5]
    ref_on_map_dir = ((pose[:, 2]+math.pi) * nangles / (2*math.pi)).round().long() % nangles
    #np.set_printoptions(precision=2,linewidth=250,threshold=100000)
    #print('R, Theta, X, Y, map_x, map_y, phi_head, map_dir')
    #print(torch.stack([r, t, x, y, ref_on_map_x.long().float(), ref_on_map_y.long().float(), normalized_angles*180.0/math.pi, ref_on_map_dir.float()], dim=1)[:5].detach().cpu().numpy())
    #pdb.set_trace()
    # print('convert_world2map','pose:',pose,'ref_on_map:',torch.stack([ref_on_map_x, ref_on_map_y, ref_on_map_dir], dim=1),sep='\n')
    return torch.stack([ref_on_map_x, ref_on_map_y, ref_on_map_dir], dim=1)

def convert_map2world_ori(pose, map_shape, map_scale, angles):
    """
    Inputs:
        pose - (bs, 3) Tensor agent pose in map coordinates (x, y, theta) 
             - Note: theta is discrete angle idx
        map_shape - (f, h, w) tuple
        map_scale - scalar
        angles - (nangles, ) Tensor

    Conventions:
        world positions - X rightward, Y upward, origin at center
        map positions - X rightward, Y downward, origin at top-left
    """
    x = pose[:, 0].float()
    y = pose[:, 1].float()
    angle_idx = pose[:, 2].long()
    mh, mw = map_shape[1:]

    x_world = (x - (mw-1)/2) * map_scale
    y_world = ((mh-1)/2 - y) * map_scale
    theta_world = angles[angle_idx]
    # print('convert_map2world','ref_on_map:',pose,'pose:',torch.stack([x_world, y_world, theta_world], dim=1),sep='\n')
    return torch.stack([x_world, y_world, theta_world], dim=1)

def convert_world2map(pose, map_shape, map_scale, angles):
    """
    Inputs:
        pose - (bs, 3) Tensor agent pose in world coordinates (x, y, theta) [0.5,9.5] [0.5,9.5] [0,2pi]
        map_shape - (f, h, w) tuple     32x31x31
        map_scale - scalar              1
        angles - (nangles, ) Tensor     [0,2pi) dim=12
        eps - scalar angular bin-size   

    Conventions:
        Relative world positions - X downward, Y rightward, origin at center
        map positions - X downward, Y rightward, origin at top-left
    """
    x = pose[:, 0]
    y = pose[:, 1]
    mh, mw = map_shape[1:]
    nangles = float(angles.shape[0])
    # This convention comes from transform_to_map() in model_pose.py [0,100] when mw=101 
    ref_on_map_x = torch.clamp((mw-1)/2 + x*map_scale, 0, mw-1).round().long() 
    ref_on_map_y = torch.clamp((mh-1)/2 + y*map_scale, 0, mh-1).round().long()
    # Mapping heading angles to map locations [6,7,8,9,10,11,0,1,2,3,4,5]
    ref_on_map_dir = (((pose[:, 2]) * nangles / (2*math.pi)).round() % nangles).long()
    return torch.stack([ref_on_map_x, ref_on_map_y, ref_on_map_dir], dim=1)

def convert_map2world(pose, map_shape, map_scale, angles):
    """
    Inputs:
        pose - (bs, 3) Tensor agent pose in map coordinates (x, y, theta) 
             - Note: theta is discrete angle idx
        map_shape - (f, h, w) tuple
        map_scale - scalar
        angles - (nangles, ) Tensor

    Conventions:
        Relative world positions - X rightward, Y upward, origin at center
        map positions - X downward, Y rightward, origin at top-left
    """
    x = pose[:, 0].float()
    y = pose[:, 1].float()
    angle_idx = pose[:, 2].long()
    mh, mw = map_shape[1:]

    x_world = (x - (mw-1)/2) / map_scale
    y_world = (y - (mh-1)/2) / map_scale
    theta_world = angles[angle_idx]
    # print('convert_map2world','ref_on_map:',pose,'pose:',torch.stack([x_world, y_world, theta_world], dim=1),sep='\n')
    return torch.stack([x_world, y_world, theta_world], dim=1)




def convert_polar2xyt(poses):
    """
    poses - (bs, 3) torch Tensor of (r, phi, theta) poses
    converts to (x, y, theta)
    """
    r, phi, theta = poses[:, 0], poses[:, 1], poses[:, 2]
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    poses_xyz = torch.stack([x, y, theta], dim=1)
    return poses_xyz
    
def compute_relative_pose_ori(pose_a, pose_b):
    """
    Compute pose of pose_b in the egocentric coordinate frame of pose_a.
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)
    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = pose_a[:, 0], pose_a[:, 1], pose_a[:, 2]
    x_b, y_b, theta_b = pose_b[:, 0], pose_b[:, 1], pose_b[:, 2]

    r_ab = torch.sqrt((x_a - x_b)**2 + (y_a - y_b)**2) # (bs, )
    phi_ab = torch.atan2(y_b - y_a, x_b - x_a) - theta_a # (bs, )
    theta_ab = theta_b - theta_a # (bs, )
    theta_ab = torch.atan2(torch.sin(theta_ab), torch.cos(theta_ab))

    x_ab = torch.stack([
        r_ab * torch.cos(phi_ab),
        r_ab * torch.sin(phi_ab),
        theta_ab,
    ], dim=1) # (bs, 3)

    return x_ab

def compute_relative_pose(pose_a, pose_b):
    """
    Compute pose of pose_b in the egocentric coordinate frame of pose_a.
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)
    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = pose_a[:, 0], pose_a[:, 1], pose_a[:, 2]
    x_b, y_b, theta_b = pose_b[:, 0], pose_b[:, 1], pose_b[:, 2]

    r_ab = torch.sqrt((x_a - x_b)**2 + (y_a - y_b)**2) # (bs, )
    phi_ab = torch.atan2(y_b - y_a, x_b - x_a) # (bs, )
    theta_ab = theta_b - theta_a # (bs, )
    theta_ab = torch.atan2(torch.sin(theta_ab), torch.cos(theta_ab))

    x_ab = torch.stack([
        r_ab * torch.cos(phi_ab),
        r_ab * torch.sin(phi_ab),
        theta_ab,
    ], dim=1) # (bs, 3)

    return x_ab


def process_maze_batch(batch, device):
    for k in batch.keys():
        # Convert (bs, L, ...) -> (L, bs, ...)
        batch[k] = batch[k].transpose(0, 1).contiguous().to(device).float()
    # Rotate image by 90 degrees counter-clockwise --- agent is facing upward
    batch['rgb'] = torch.flip(batch['rgb'].transpose(3, 4), [3]) # (bs, L, 2, H, W)
    # Converting to world coordinates convention
    x = -batch['poses'][..., 1]
    y = batch['poses'][..., 0]
    t = batch['poses'][..., 2]
    batch['poses'] = torch.stack([x, y, t], dim=2)
    return batch

def save_trajectory(inputs,labels,path,angles,name='train'):
    '''
    inputs:L-1,bs,nangles,H,W
    labels:L-1,bs,3
    '''
    L, bs, nangles, H, W = inputs.shape
    inputs = asnumpy(inputs)
    labels = asnumpy(labels)
    angles = angles.cpu()
    pred_pos = np.unravel_index(np.argmax(inputs.reshape(L*bs, -1), axis=1), inputs.shape[1:])
    pred_pos = np.stack(pred_pos, axis=1) # (L*bs, 3) shape with (theta_idx, y, x)
    pred_pos = np.ascontiguousarray(np.flip(pred_pos, axis=1)) # Convert to (x, y, theta_idx)
    pred_pos = torch.Tensor(pred_pos).long() # (L*bs, 3) --> (x, y, dir)
    pred_world_pos = convert_map2world(pred_pos, (nangles, H, W), map_scale=1, angles=angles) # (L*bs, 3) --> (x_world, y_world, theta_world)
    pred_world_pos = rearrange(asnumpy(pred_world_pos), '(t b) n -> t b n',t=L)
    np.savez(os.path.join(path,name),pred=pred_world_pos, gt=labels)

class OUActionNoise(object):
    def __init__(self, mu, sigma=1, theta=0.2, dt=0.01, x0=None):
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

def imu_world2map(pose, map_scale, angles):

    x = pose[:, 0]
    y = pose[:, 1]

    nangles = float(angles.shape[0])
    # This convention comes from transform_to_map() in model_pose.py [0,100] when mw=101 
    ref_on_map_x = (x/map_scale).round().long() 
    ref_on_map_y = (y/map_scale).round().long() 
    # Mapping heading angles to map locations [6,7,8,9,10,11,0,1,2,3,4,5]
    ref_on_map_dir = ((pose[:, 2]) * nangles / (2*math.pi)).round().long() % nangles
    return torch.stack([ref_on_map_x, ref_on_map_y, ref_on_map_dir], dim=1)

def rotate_tensor(r, t, mode, pad_mode="zeros"):
    """
    rotate clockwise

    Inputs:
        r     - (h, w) Tensor
        t     - 1 Tensor of angles
    Outputs:
        r_rot - (h, w) Tensor
    """
    r = torch.tensor(r[None,None,...]).float() #(1, 1, h, w) 
    t = torch.tensor(t[None,]).float() # (1, )
    sin_t  = torch.sin(t)
    cos_t  = torch.cos(t)
    # This R convention means Y axis is downwards.
    A      = torch.zeros(r.size(0), 2, 3)
    # rotate clockwise
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = sin_t
    A[:, 1, 0] = -sin_t
    A[:, 1, 1] = cos_t
    # # rotate counter clockwise
    # A[:, 0, 0] = cos_t
    # A[:, 0, 1] = -sin_t
    # A[:, 1, 0] = sin_t
    # A[:, 1, 1] = cos_t
    grid   = F.affine_grid(A, r.size(), align_corners=False)
    r_rot  = F.grid_sample(r, grid,mode=mode, padding_mode=pad_mode, align_corners=False)
    return r_rot[0,0].numpy()

# def rotate_tensor(r, t, mode='bilinear'):
#     """
#     Inputs:
#         r     - (bs, f, h, w) Tensor
#         t     - (bs, ) Tensor of angles
#     Outputs:
#         r_rot - (bs, f, h, w) Tensor
#     """
#     device = r.device
#     sin_t  = torch.sin(t)
#     cos_t  = torch.cos(t)

#     # This R convention means Y axis is downwards.
#     A      = torch.zeros(r.size(0), 2, 3).to(device)
#     # rotate counter clockwise
#     A[:, 0, 0] = cos_t
#     A[:, 0, 1] = -sin_t
#     A[:, 1, 0] = sin_t
#     A[:, 1, 1] = cos_t

#     grid   = F.affine_grid(A, r.size(), align_corners=False)
#     r_rot  = F.grid_sample(r, grid,mode=mode, align_corners=False)
#     return r_rot

def rotation_resample(x, angles, mode='bilinear', offset=0.0):
    """
    Inputs:
        x       - (bs, f, s, s) feature maps
        angles_ - (nangles, ) set of angles to sample
    Outputs:
        x_rot   - (bs, nangles, f, s, s)
    """
    # angles      = angles_.clone() # (nangles, )
    bs, f, s, s = x.shape
    nangles     = angles.shape[0]
    x_rep       = x.unsqueeze(1).expand(-1, nangles, -1, -1, -1) # (bs, nangles, f, s, s)
    x_rep       = rearrange(x_rep, 'b o e h w -> (b o) e h w')
    angles      = angles.unsqueeze(0).expand(bs, -1).contiguous().view(-1) # (bs * nangles, )
    x_rot       = rotate_tensor(x_rep, angles + math.radians(offset),mode) # (bs * nangles, f, s, s)
    x_rot       = rearrange(x_rot, '(b o) e h w -> b o e h w', b=bs)

    return x_rot

def allocentric_poses(ego_poses):
    '''
        input: [L,3] x, y, yaw
            where each pose is relative its last pose

        output:[L,3]
            where each pose is relative to start pose
    '''
    # Initialize the allocentric pose as the first pose in the sequence
    allo_pose = ego_poses[0]
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
    return output

def egocentric_poses(allo_poses):
    # Initialize the previous allocentric pose as the first pose in the sequence
    prev_allo_pose = allo_poses[0]
    # Initialize an empty list to store the output poses
    output = [prev_allo_pose]
    # Loop through the rest of the poses in the sequence
    for allo_pose in allo_poses[1:]:
        # Calculate the egocentric position and orientation of the current pose
        ego_x = (allo_pose[0] - prev_allo_pose[0]) * math.cos(prev_allo_pose[2]) + (allo_pose[1] - prev_allo_pose[1]) * math.sin(prev_allo_pose[2])
        ego_y = -(allo_pose[0] - prev_allo_pose[0]) * math.sin(prev_allo_pose[2]) + (allo_pose[1] - prev_allo_pose[1]) * math.cos(prev_allo_pose[2])
        ego_theta = (allo_pose[2] - prev_allo_pose[2]) % (2 * math.pi)
        if ego_theta > math.pi:
            ego_theta -= 2 * math.pi
        elif ego_theta < -math.pi:
            ego_theta += 2 * math.pi
        # Update the previous allocentric pose with the current pose
        prev_allo_pose = allo_pose
        # Append the egocentric pose to the output list
        output.append([ego_x, ego_y, ego_theta])
        # Return the output list of egocentric poses
    return output

def layer0_mask(h,w):
    mask = np.zeros((h,w))
    row, col = h // 2, w // 2
    j = 0
    for i in range(row, h):
        l, r = col-j, col+j+1
        mask[i,l:r] = 1
        j += 1
    return mask

def mask_rotate(mask,angle,position,H=33,W=33,h=16,w=16):
    def center_observation(observation,center,x,y):
        left,right,up,down = center[1], len(observation[0])-center[1]-1, center[0], len(observation)-center[0]-1
        # print(left,right,up,down)
        observation = np.hstack((np.zeros((len(observation),y-left)),observation)) if left<y else observation if left==y else observation[:,(left-y):]
        observation = np.hstack((observation,np.zeros((len(observation),y-right)))) if right < y else observation  if right == y else observation[:,:(y-right)]
        observation = np.vstack((np.zeros((x-up,len(observation[0]))),observation)) if up < x else observation if up ==x else observation[up-x:,:]
        observation = np.vstack((observation,np.zeros((x-down,len(observation[0]))))) if down < x else observation if down==x else observation[:(x-down),:]
        return observation
    # print(mask)
    # print(f'degree:{angle}')
    # mask1 = rotate(mask, angle=angle,order=1,reshape=False,mode='constant')
    mask = rotate_tensor(mask, -angle, 'bilinear')
    # print(f'rotate_mask:\n{mask1}')
    # print(f'torch_rotate_mask:\n{mask2}')
    # print(f'compare:\n{abs(mask1-mask2)<0.1}')
    # mask = center_observation(mask,[H-1-position[0],W-1-position[1]],h,w)
    # print(mask)
    return mask

def test_localize():
    rot_obs = torch.arange(2*3*4*5*5).view(2,3,4,5,5).float() #b,nangles,cls,h,w
    pose = torch.zeros(2,3,5,5).float() #b,nangles,h,w
    pose[0,1,2,4] = 1
    pose[1,0,3,3] = 1
    print(f'pose:{pose.shape}\n{pose}')
    print(f'obs:{rot_obs.shape}\n{rot_obs}')
    rot_obs = rearrange(rot_obs,'b o e h w -> (b o) e h w')
    pose = rearrange(pose, 'b o h w -> () (b o) h w')
    o_reg = F.conv_transpose2d(pose, rot_obs, groups=2, padding=2) # (1, bs*f, h, w)
    o_reg = rearrange(o_reg, '() (b e) h w -> b e h w', b=2)
    print(f'allo:{o_reg.shape}\n{o_reg}')

if __name__ == "__main__":
    # def test_centric_convert():
    #     a1 = torch.tensor([1,1,0.5236])
    #     a2 = torch.tensor([2,2,1.0472])
    #     a3 = torch.tensor([3,4,-2.0944])
    #     c1 = compute_relative_pose(a1[None,],a2[None,])
    #     c2 = compute_relative_pose(a2[None,],a3[None,])
    #     print(f'a1:{a1}\na2:{a2}\na3:{a3}\nc1:{c1}\nc2:{c2}')
    #     c3 = egocentric_poses([a1,a2,a3])
    #     c4 = allocentric_poses([a1]+c3)
    #     print(f'c3:{c3}\nc4:{c4}')

    #     return
    # test_centric_convert()
    test_localize()
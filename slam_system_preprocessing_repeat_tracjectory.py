import os,sys,copy,glob,time,math,torch,cv2,scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from einops import rearrange, reduce, asnumpy, parse_shape
from scipy.ndimage import rotate
import torch.nn.functional as F
from einops import rearrange, reduce, asnumpy, parse_shape
from functools import partial
from datetime import date
import sys
from tqdm import tqdm
import argparse
import pprint as pp
from utils import *
import random
'''
first_release:
    Sep19th 2022

    April6th 2023
        Samples a random set of batch_size episodes. 
        Input: 
            rgb:            (L,B,3,h,w)
            depth:          (L,B,3,h,w)
            image_cls:      (L,B,cls,local_mapsize,local_mapsize)
            delta:          (L,B,3)
            maps:           (L,B,cls,mapsize,mapsize) observabale maps to current step
            map_labl:       (B,mapsize,mapsize)
            map_cls_labl:   (B,cls,mapsize,mapsize)   convert map_label value to onehot tensor
        Outputs:
            episode - generator with each output as 
                      dictionary, values are torch Tensor observations
                      {
                          rgb:          L,B,3,h,w
                          depth:        (L,B,3,h,w)
                          image_cls :     L,B,cls,C,H,W
                          poses:    L,B,3  (x,y,yaw)
                          maps:     L,B,cls,mapsize,mapsize
                          map_labl: same as input
                          map_cls_labl: same as input
                          pose_changes: L-1,B,3
                          gt_poses: L-1,B,3 Long dtype in tensor coordinate
                      }       
        '''
torch.set_printoptions(linewidth=4000,precision=3,threshold=1000000,sci_mode=False)
np.set_printoptions(linewidth=10000,threshold=100000,precision=3,suppress=True)
np.random.seed(0)
torch.manual_seed(0)
random.seed(0) 

torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='Feb15th_baseline')   
    parser.add_argument('--map_size', type=int, default=11,help='The size of environment')
    parser.add_argument('--obs_size', type=int, default=11,help='The size of environment')
    parser.add_argument('--n_object', type=int, default=21,help='Item quantities')
    parser.add_argument('--feature_dim', type=int, default=11, help='Item varieties + 1 ground per environment')
    parser.add_argument('--env_ctr', type=int, default=30, help='total environemtns')
    # parser.add_argument('--tra_ctr', type=int, default=30, help='number of trajecotry per environment')
    parser.add_argument('--train_env', type=int, default=28, help='environment to train')
    parser.add_argument('--same_env', type=int, default=1, help='environment to train')
    parser.add_argument('--rot_mode', type=str, default='bilinear', help='bilinear or nearest')  

    #env_setting
    parser.add_argument('--tra_repeat', type=int, default=2, help='times to repeat the trajectory')
    parser.add_argument('--num_steps', type=int, default=31)
    parser.add_argument('--angles_intvl', type=int, default=45)
    parser.add_argument('--map_scale', type=int, default=1)


    args = parser.parse_args()
    args.noise = [0.2,0.15,0.01]
    args.map_shape = (args.feature_dim, args.map_size, args.map_size)
    args.angles = np.radians(np.linspace(0, int(360-args.angles_intvl), int(360//args.angles_intvl)))
    args.map_config = [args.map_shape,args.map_scale,args.angles]

    args.out_path = f'/data1/mli170/2022_Sep20th_dataset/{args.date}_repeat{args.tra_repeat}_m{args.map_size}obj{args.n_object}len{args.num_steps}_angl{args.angles_intvl}_{args.env_ctr}env/model'
    args.train_path = 'oldData/Gazebo_Feb15th_28of30env_map11_obj21_len40_w_rgbd_train_small.npz'
    args.val_path = 'oldData/Gazebo_Feb15th_2of30env_map11_obj21_len40_w_rgbd_test_small.npz'
    # args.train_path = 'Gazebo_Feb15th_realYolo_scale3_28of30env_map33_obj21_len61_wo_rgbd_train_small.npz'
    # args.val_path = 'Gazebo_Feb15th_realYolo_scale3_2of30env_map33_obj21_len61_wo_rgbd_test_small.npz'
    # args.train_path = 'Gazebo_Feb15th_perfect_backend_scale3_28of30env_map33_obj21_len40_train.npz'
    # args.val_path = 'Gazebo_Feb15th_perfect_backend_scale3_2of30env_map33_obj21_len40_test.npz'
    today = date.today()
    print(today.strftime("%b-%d-%Y"))
    print(sys.argv)
    pp.pprint(vars(args))
    return args

class preprocess():
    def __init__(self):
        args = get_args()
        self.num_steps = args.num_steps
        self.args = args
        self.n_cls = args.feature_dim
        self.device='cuda:0'
        # self.device='cpu'
        self.data_idx = 0
        self.map_shape, self.map_scale, self.angles = args.map_config
        self.tra_repeat = args.tra_repeat
        self.extended_steps = self.num_steps*self.tra_repeat
        
    def generate(self, key='train'):
        'L,b,... or b,...'

        if key == 'train':
            data_file = np.load(self.args.train_path,allow_pickle=True)
            out_path = os.path.join(self.args.out_path,'train')
            print('\n\nGenerate training file')
        else:
            data_file = np.load(self.args.val_path,allow_pickle=True)    
            print('\n\nGenerate evaluating file')
            out_path = os.path.join(self.args.out_path,'val')
        # try:
        #     os.makedirs(out_path)
        # except OSError:
        #     pass
        self.obs_keys = list(data_file.files)
        self.nsplit = data_file[self.obs_keys[2]].shape[1] # get total batch
        self.max_steps = data_file[self.obs_keys[2]].shape[0]
        print(f'number of files:{self.nsplit}')
        print(f'number of steps in one trajectory:{self.max_steps}')
        
        for index in tqdm(range(self.nsplit)):
            self.episodes = {}
            max_batch = data_file[self.obs_keys[2]].shape[1]
            for key_ in self.obs_keys:
                # print(f'{key_}:{data_file[key_].shape}')
                # if key_ in ['rgb','depth']: continue
                if key in ['image_cls','maps','map_labl','map_cls_labl','semnatic_gt_poses']:continue
                if 'labl' not in key_:
                    dimsize = data_file[key_][:,index].ndim
                    repeat = [1] * dimsize
                    repeat[0] = self.tra_repeat
                    self.episodes[key_] = torch.tensor(data_file[key_][:,index][:self.num_steps],device=self.device).float().repeat(repeats=repeat)
                else:
                    self.episodes[key_] = torch.tensor(data_file[key_][index],device=self.device).float()


            #preprocessing image
            # angles = torch.Tensor(np.radians(np.linspace(0, 359, int(360//self.args.angles_intvl)))).to(self.device)
            # self.episodes['image_cls'] = rotation_resample(self.episodes['image_cls'], angles)


            # poses = torch.zeros(self.num_steps, 3, device=self.device) #pose:(L,3)
            poses = self.episodes['delta']
            # for t in range(self.num_steps):
            #     # poses[t] = poses[t-1] + convert_polar2xyt(episodes['delta'][t]) #convert polar to xyt
            #     poses[t] = self.episodes['delta'][t] # x_y_yaw : in world coordinate
            # self.episodes['poses']=poses #L-1,3
            self.episodes['poses'] = self.episodes['delta']
            obs_poses = poses.clone()
            imu_pos_ori = poses.clone()
            imu_poses = poses.clone()
            start_pose = poses[0].clone()
            # imu_ego = torch.tensor(egocentric_poses(poses.clone().numpy()))
            # imu_back_allo = torch.tensor(allocentric_poses(imu_ego.clone().numpy()))
            for l in range(self.extended_steps):
                obs_poses[l] = compute_relative_pose_ori(start_pose[None,:], obs_poses[l][None,:])[0]
                
                if l > 0:
                    # imu_poses[l] = compute_relative_pose(poses[l-1][None,:], poses[l][None,:])[0]
                    imu_pos_ori[l] = compute_relative_pose_ori(poses[l-1][None,:], poses[l][None,:])[0]
                    # print(f'step:{l}\nimu:\t{imu_poses[l]}\nimu_ori:{imu_pos_ori[l]}\nimu_ego:{imu_ego[l]}\npose:{poses[l]}\nback:{imu_back_allo[l]}')
            self.episodes['pose_changes']=obs_poses[1:] #L-1,bs,3 where first pose is center
            self.episodes['imu']=imu_pos_ori[1:]
            # convert world poses to tensor map index
            gt_poses = convert_world2map(
                obs_poses,
                self.map_shape,
                self.map_scale, 
                self.angles,
            ).long()
            self.episodes['gt_poses'] = gt_poses[1:] # (L-1,3) 

            index_pose = gt_poses[1:]
            semnatic_gt_poses = torch.zeros(self.extended_steps-1, self.angles.shape[0], self.args.map_size, self.args.map_size, device=self.device)  #((L-1)*bs,nangles, mapsize,mapsize)
            semnatic_gt_poses[range(int(self.extended_steps-1)) ,index_pose[:, 2], index_pose[:, 0], index_pose[:, 1]] = 1.0
            self.episodes['semnatic_gt_poses'] = semnatic_gt_poses # (L-1,3) 


            # index_pose = rearrange(gt_poses[1:],'t b c -> (t b) c ')
            # semnatic_gt_poses = rearrange(torch.zeros(self.num_steps-1, max_batch, self.args.nangles, self.args.map_size, self.args.map_size, device=self.device), 't b c h w -> (t b) c h w')  #((L-1)*bs,nangles, mapsize,mapsize)
            # semnatic_gt_poses[range(int((self.num_steps-1)*max_batch)) ,index_pose[:, 2], index_pose[:, 0], index_pose[:, 1]] = 1.0
            # semnatic_gt_poses = rearrange(semnatic_gt_poses,'(t b) c h w -> t b c h w', t = self.num_steps-1)
            # self.episodes['semnatic_gt_poses'] = semnatic_gt_poses # (L-1,b,nangles,h,w) 
            for key_ in self.episodes.keys():
                self.episodes[key_] = self.episodes[key_].cpu().numpy()

            # for model in ['semantic']:
            for model in ['mapnet', 'deepvo']:
            # for model in ['mapnet', 'deepvo','semantic']:
                out_model_path = out_path.replace('model',model)
                try:
                    os.makedirs(out_model_path)
                except OSError:
                    pass
                if model == 'mapnet':
                    np.savez(f'{out_model_path}/{index}',
                    rgb = self.episodes['rgb'],
                    depth = self.episodes['depth'],
                    pose_changes=self.episodes['pose_changes'], #(l-1,bs,3)
                    gt_poses=self.episodes['gt_poses'], #(l-1,bs,3) in tensor coordinate
                    )

                elif model == 'deepvo':  
                    np.savez(f'{out_model_path}/{index}',
                    rgb = self.episodes['rgb'],
                    pose_changes=self.episodes['pose_changes'],#(l-1,bs,3)
                    imu = self.episodes['imu'] #(l-1,bs,3)
                    )

                elif model == 'semantic':
                    np.savez(f'{out_model_path}/{index}',
                        # rgb = self.episodes['rgb'],
                        # depth = self.episodes['depth'],
                        # poses=self.episodes['poses'],
                        # delta=self.episodes['delta'], 
                        image_cls=self.episodes['image_cls'], 
                        maps=self.episodes['maps'], 
                        map_labl=self.episodes['map_labl'], 
                        map_cls_labl=self.episodes['map_cls_labl'],
                        pose_changes=self.episodes['pose_changes'],
                        gt_poses=self.episodes['gt_poses'],
                        semnatic_gt_poses=self.episodes['semnatic_gt_poses'],
                        imu = self.episodes['imu']
                        )
                else:
                    print('error')

            # print(self.episodes['delta'])
            # print(f'imu:\n{imu_poses}')
            # print(f'obs_poses:\n{obs_poses}')
            # print(f'image_cls:\n',self.episodes['image_cls'])
            # print(f'maps:\n',self.episodes['maps'])
            # print(f'gt_poses:\n',self.episodes['gt_poses'])
        for key_ in self.episodes.keys():
            print(f'{key_}:{self.episodes[key_].shape}')  
        for key_ in self.episodes.keys():  
            print(f'5steps:{key_}:{self.episodes[key_][:5]}')  
if __name__ == '__main__':
    a = preprocess()
    a.generate('train')
    a.generate('val')
        
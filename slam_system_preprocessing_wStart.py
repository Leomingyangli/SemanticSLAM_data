import os,sys,copy,glob,time,math,torch,cv2,scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
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
torch.set_printoptions(linewidth=4000,precision=5,threshold=1000000,sci_mode=False)
np.set_printoptions(linewidth=10000,threshold=100000,precision=3,suppress=True)
np.random.seed(0)
torch.manual_seed(0)
random.seed(0) 

torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_args():
    new_directory = "/home/mli170/SLAM_PROJECT/SemanticSLAM_data"  # Replace this with the desired directory path
    os.chdir(new_directory)
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='Aug20th_ceil1floor02_resnet_CrossScene_addStart')  
    parser.add_argument('--map_size', type=int, default=11,help='The size of environment')
    parser.add_argument('--obs_size', type=int, default=11,help='The size of environment')
    parser.add_argument('--n_object', type=int, default=41,help='Item quantities')
    parser.add_argument('--feature_dim', type=int, default=11, help='Item varieties + 1 ground per environment')
    parser.add_argument('--env_ctr', type=int, default=30, help='total environemtns')
    # parser.add_argument('--tra_ctr', type=int, default=30, help='number of trajecotry per environment')
    parser.add_argument('--train_env', type=int, default=26, help='environment to train')
    parser.add_argument('--same_env', type=int, default=1, help='environment to train')
    parser.add_argument('--rot_mode', type=str, default='bilinear', help='bilinear or nearest')  

    #env_setting
    parser.add_argument('--num_steps', type=int, default=80)
    parser.add_argument('--angles_intvl', type=int, default=45)
    parser.add_argument('--map_scale', type=int, default=3)


    args = parser.parse_args()
    # args.output_folder = 'May6th_YoloSegment'
    # args.output_folder = 'Apr15th_realYolo'
    # args.output_folder = 'Jun6th_YoloSegment'
    args.input_folder = 'data_raw'
    args.output_folder = 'data_pprc'

    args.map_size *= int(args.map_scale)
    args.obs_size *= int(args.obs_size)
    args.noise = [0.2,0.15,0.01]
    args.map_shape = (args.feature_dim, args.map_size, args.map_size)
    args.angles = np.radians(np.linspace(0, int(360-args.angles_intvl), int(360//args.angles_intvl)))
    args.map_config = [args.map_shape,args.map_scale,args.angles]

    args.train_path = f'{args.input_folder}/Gazebo_Aug20th_ceil1floor02_resnet_scale3_CrossScene_map33_obj40_len80_train.npz'
    args.val_path = f'{args.input_folder}/Gazebo_Aug20th_ceil1floor02_resnet_scale3_CrossScene_map33_obj40_len80_test.npz'


    # args.out_path = f'/data1/mli170/2022_Sep20th_dataset/{args.output_folder}/{args.date}_m{args.map_size}obj{args.n_object}len{args.num_steps}angl{args.angles_intvl}/semantic'
    args.out_path = f'{args.output_folder}/{args.date}_m{args.map_size}obj{args.n_object}len{args.num_steps}angl{args.angles_intvl}/semantic'
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
        self.device='cpu'
        self.data_idx = 0
        self.map_shape, self.map_scale, self.angles = args.map_config
        
    def generate(self, key='train'):
        'L,b,... or b,...'
        self.mode = key
        if key == 'train':
            data_file = np.load(self.args.train_path,allow_pickle=True)
            out_path = os.path.join(self.args.out_path,'obs_mode','train')
            print('\n\nGenerate training file')
        else:
            data_file = np.load(self.args.val_path,allow_pickle=True)    
            print('\n\nGenerate evaluating file')
            out_path = os.path.join(self.args.out_path,'obs_mode','val')

        self.obs_keys = list(data_file.files)
        self.nsplit = data_file[self.obs_keys[2]].shape[1] # get total batch
        self.max_steps = data_file[self.obs_keys[2]].shape[0]
        print(f'number of files:{self.nsplit}')
        print(f'number of steps in one trajectory:{self.max_steps}')
        
        for index in tqdm(range(self.nsplit)):
            self.episodes = {}
            # max_batch = data_file[self.obs_keys[0]].shape[1]
            for key_ in self.obs_keys:
                # print(f'{key_}:{data_file[key_].shape}')
                if key_ in ['rgb','depth']: continue
                if 'labl' not in key_:
                    self.episodes[key_] = torch.tensor(data_file[key_][:,index][:self.num_steps],device=self.device).float()
                    
                else:
                    self.episodes[key_] = torch.tensor(data_file[key_][index],device=self.device).float()

            start_map = torch.zeros(self.episodes['maps'].shape[1:])[None,...] #(1,c,h,w)
            self.episodes['maps'] = torch.concat([start_map, self.episodes['maps']], dim=0) #(l+1,c,h,w)


            #preprocessing image
            # angles = torch.Tensor(np.radians(np.linspace(0, 359, int(360//self.args.angles_intvl)))).to(self.device)
            # self.episodes['image_cls'] = rotation_resample(self.episodes['image_cls'], angles)
            # self.episodes['rgb'] = F.interpolate(self.episodes['rgb'],scale_factor=0.25,mode='bilinear')
            # self.episodes['depth'] = F.interpolate(self.episodes['depth'],scale_factor=0.25,mode='nearest')

            # self.episodes['rgb'] = unflatten_two(process_image(flatten_two(self.episodes['rgb'][:,None]),self.mode), self.num_steps, 1)[:,0]
            
            # print("self.episodes['rgb']:",self.episodes['rgb'].shape)
            # poses = torch.zeros(self.num_steps, 3, device=self.device) #pose:(L,3)
            poses = self.episodes['delta']
            self.episodes['poses'] = self.episodes['delta']
            obs_poses = poses.clone()
            start_pose = poses[0].clone()

            imu_pos_ori = poses.clone()
            imu_poses = poses.clone()
            

            for l in range(self.num_steps):
                obs_poses[l] = compute_relative_pose_ori(start_pose[None,:], obs_poses[l][None,:])[0]
                imu_poses[l] = compute_relative_pose(start_pose[None,:], poses[l][None,:])[0]
                imu_pos_ori[l] = compute_relative_pose_ori(start_pose[None,:], poses[l][None,:])[0]
                if l > 0:
                    imu_poses[l] = compute_relative_pose(poses[l-1][None,:], poses[l][None,:])[0]
                    imu_pos_ori[l] = compute_relative_pose_ori(poses[l-1][None,:], poses[l][None,:])[0]
                    # print(f'step:{l}\nimu:\t{imu_poses[l]}\nimu_ori:{imu_pos_ori[l]}\nimu_ego:{imu_ego[l]}\npose:{poses[l]}\nback:{imu_back_allo[l]}')
            self.episodes['pose_changes']=obs_poses #L,bs,3 where first pose is center
            self.episodes['imu']=imu_pos_ori #L,bs,3 where imu is 0
            # convert world poses to tensor map index
            gt_poses = convert_world2map(
                obs_poses,
                self.map_shape,
                self.map_scale, 
                self.angles,
            ).long()
            self.episodes['gt_poses'] = gt_poses # (L,3) 

            semnatic_gt_poses = torch.zeros(self.num_steps, self.angles.shape[0], self.args.map_size, self.args.map_size, device=self.device)  #((L-1)*bs,nangles, mapsize,mapsize)
            semnatic_gt_poses[range(int(self.num_steps)) ,gt_poses[:, 2], gt_poses[:, 0], gt_poses[:, 1]] = 1.0
            self.episodes['semnatic_gt_poses'] = semnatic_gt_poses # (L,3)
            
            # index_pose = rearrange(gt_poses[1:],'t b c -> (t b) c ')
            # semnatic_gt_poses = rearrange(torch.zeros(self.num_steps-1, max_batch, self.args.nangles, self.args.map_size, self.args.map_size, device=self.device), 't b c h w -> (t b) c h w')  #((L-1)*bs,nangles, mapsize,mapsize)
            # semnatic_gt_poses[range(int((self.num_steps-1)*max_batch)) ,index_pose[:, 2], index_pose[:, 0], index_pose[:, 1]] = 1.0
            # semnatic_gt_poses = rearrange(semnatic_gt_poses,'(t b) c h w -> t b c h w', t = self.num_steps-1)
            # self.episodes['semnatic_gt_poses'] = semnatic_gt_poses # (L-1,b,nangles,h,w) 
            for key_ in self.episodes.keys():
                self.episodes[key_] = self.episodes[key_].cpu().numpy()

            for model in ['real_obs']:
            # for model in ['real_obs', 'obst_obs', 'perf_obs']:
            # for model in ['rgbd']:
            # for model in ['mapnet', 'deepvo','semantic']:
                out_model_path = out_path.replace('obs_mode',model)
                # out_model_path = os.path.join(out_path, model)
                try:
                    os.makedirs(out_model_path)
                except OSError:
                    pass
                if model == 'real_obs':
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
                elif model == 'obst_obs':
                    np.savez(f'{out_model_path}/{index}',
                        # rgb = self.episodes['rgb'],
                        # depth = self.episodes['depth'],
                        # poses=self.episodes['poses'],
                        # delta=self.episodes['delta'], 
                        image_cls=self.episodes['image_cls_obst'], 
                        maps=self.episodes['maps_obst'], 
                        map_labl=self.episodes['map_labl'], 
                        map_cls_labl=self.episodes['map_cls_labl'],
                        pose_changes=self.episodes['pose_changes'],
                        gt_poses=self.episodes['gt_poses'],
                        semnatic_gt_poses=self.episodes['semnatic_gt_poses'],
                        imu = self.episodes['imu']
                        )
                elif model == 'perf_obs':
                    np.savez(f'{out_model_path}/{index}',
                        # rgb = self.episodes['rgb'],
                        # depth = self.episodes['depth'],
                        # poses=self.episodes['poses'],
                        # delta=self.episodes['delta'], 
                        image_cls=self.episodes['image_cls_perf'], 
                        maps=self.episodes['maps_perf'], 
                        map_labl=self.episodes['map_labl'], 
                        map_cls_labl=self.episodes['map_cls_labl'],
                        pose_changes=self.episodes['pose_changes'],
                        gt_poses=self.episodes['gt_poses'],
                        semnatic_gt_poses=self.episodes['semnatic_gt_poses'],
                        imu = self.episodes['imu']
                        )
                elif model == 'rgbd':
                    np.savez(f'{out_model_path}/{index}',
                        rgb = self.episodes['rgb'],
                        depth = self.episodes['depth'],
                        poses=self.episodes['poses'],
                        # delta=self.episodes['delta'], 
                        # image_cls=self.episodes['image_cls_perf'], 
                        # maps=self.episodes['maps_perf'], 
                        # map_labl=self.episodes['map_labl'], 
                        # map_cls_labl=self.episodes['map_cls_labl'],
                        pose_changes=self.episodes['pose_changes'],
                        gt_poses=self.episodes['gt_poses'],
                        # semnatic_gt_poses=self.episodes['semnatic_gt_poses'],
                        imu = self.episodes['imu']
                        )
                else:
                    print('error')
                
                # if model == 'mapnet':
                #     np.savez(f'{out_model_path}/{index}',
                #     rgb = self.episodes['rgb'],
                #     depth = self.episodes['depth'],
                #     pose_changes=self.episodes['pose_changes'], #(l-1,bs,3)
                #     gt_poses=self.episodes['gt_poses'], #(l-1,bs,3) in tensor coordinate
                #     )

                # elif model == 'deepvo':  
                #     np.savez(f'{out_model_path}/{index}',
                #     rgb = self.episodes['rgb'],
                #     pose_changes=self.episodes['pose_changes'],#(l-1,bs,3)
                #     imu = self.episodes['imu'] #(l-1,bs,3)
                #     )

                # elif model == 'semantic':
                #     np.savez(f'{out_model_path}/{index}',
                #         # rgb = self.episodes['rgb'],
                #         # depth = self.episodes['depth'],
                #         # poses=self.episodes['poses'],
                #         # delta=self.episodes['delta'], 
                #         image_cls=self.episodes['image_cls'], 
                #         maps=self.episodes['maps'], 
                #         map_labl=self.episodes['map_labl'], 
                #         map_cls_labl=self.episodes['map_cls_labl'],
                #         pose_changes=self.episodes['pose_changes'],
                #         gt_poses=self.episodes['gt_poses'],
                #         semnatic_gt_poses=self.episodes['semnatic_gt_poses'],
                #         imu = self.episodes['imu']
                #         )
                # else:
                #     print('error')

            # print(self.episodes['delta'])
            # print(f'imu:\n{imu_poses}')
            # print(f'obs_poses:\n{obs_poses}')
            # print(f'image_cls:\n',self.episodes['image_cls'])
            # print(f'maps:\n',self.episodes['maps'])
            # print(f'gt_poses:\n',self.episodes['gt_poses'])
        for key_ in self.episodes.keys():
            print(f'{key_}:{self.episodes[key_].shape}')  
        for key_ in self.episodes.keys():
            for i in range(5):
                print(f'steps {i+1}:{key_}:\n{self.episodes[key_][i]}')  

def allo(ego_pose, prev_poses):
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


if __name__ == '__main__':
    a = preprocess()
    a.generate('train')
    a.generate('val')
        
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
import seaborn as sns
'''
first_release:
    Apr4th 2023
    This module is used to compare input of different datasets.
        Perfect input
        Real Yolo
        Yolo with deleteing rgb outliers whose rgb values > 250 
                      
'''
torch.set_printoptions(linewidth=4000,precision=2,threshold=1000000,sci_mode=False)
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
    parser.add_argument('--date', type=str, default='May5th_obsOverStep_olv2newPerfect')  
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
    parser.add_argument('--num_steps', type=int, default=31)
    parser.add_argument('--angles_intvl', type=int, default=45)
    parser.add_argument('--map_scale', type=int, default=3)


    args = parser.parse_args()
    args.map_size *= int(args.map_scale)
    args.obs_size *= int(args.obs_size)
    args.noise = [0.2,0.15,0.01]
    args.map_shape = (args.feature_dim, args.map_size, args.map_size)
    args.angles = np.radians(np.linspace(0, int(360-args.angles_intvl), int(360//args.angles_intvl)))
    args.map_config = [args.map_shape,args.map_scale,args.angles]
    # args.path_realYolo = 'Gazebo_Apr15th_realYolo_scale3_2of30env_map33_obj21_len40_wo_rgbd_test_small.npz'
    args.path_realYolo = 'oldData/Gazebo_Feb15th_perfect_backend_scale3_2of30env_map33_obj21_len40_test.npz'
    args.path_perfect = 'Gazebo_Apr15th_perfectYolo_scale3_2of30env_map33_obj21_len40_test.npz'
    args.path_whiteYolo = 'Gazebo_Apr30th_YoloSegment_scale3_2of30env_map33_obj21_len40_wo_rgbd_test_small.npz'
    # args.path_whiteYolo = 'Gazebo_Apr30th_YoloSegment_scale1_2of30env_map11_obj21_len40_wo_rgbd_test_small.npz'
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
        data_perfect = np.load(self.args.path_perfect,allow_pickle=True)
        data_realYolo = np.load(self.args.path_realYolo,allow_pickle=True)
        data_whiteYolo = np.load(self.args.path_whiteYolo,allow_pickle=True)

        out_path = f'printlog/yoloCompareTest/{self.args.date}'
        try:
            os.makedirs(out_path)
        except OSError:
            pass
        print('\n\nGenerate training file')


        self.obs_keys = list(data_realYolo.files)
        self.nsplit = data_realYolo[self.obs_keys[2]].shape[1] # get total batch
        self.max_steps = data_realYolo[self.obs_keys[0]].shape[0]
        print(f'number of files:{self.nsplit}')
        print(f'number of steps in one trajectory:{self.max_steps}')

        plt.ylabel('Error')
        plt.xlabel('Time steps')
        plt.grid(True,'major','y')
        for index in tqdm(range(self.nsplit)):
            print('='*20,'batch:', index, '='*20)
            self.perfect = {}
            self.realYolo = {}
            self.whiteYolo = {}
            # max_batch = data_file[self.obs_keys[0]].shape[1]
            for key_ in self.obs_keys:
                # print(f'{key_}:{data_file[key_].shape}')
                if key_ in ['rgb','depth']: continue
                if 'labl' not in key_:
                    self.perfect[key_] = torch.tensor(data_perfect[key_][:,index][:self.num_steps],device=self.device).float()
                    self.realYolo[key_] = torch.tensor(data_realYolo[key_][:,index][:self.num_steps],device=self.device).float()
                    self.whiteYolo[key_] = torch.tensor(data_whiteYolo[key_][:,index][:self.num_steps],device=self.device).float()
                else:
                    self.perfect[key_] = torch.tensor(data_perfect[key_][index],device=self.device).float()
                    self.realYolo[key_] = torch.tensor(data_realYolo[key_][index],device=self.device).float()         
                    self.whiteYolo[key_] = torch.tensor(data_whiteYolo[key_][index],device=self.device).float()
            # observation size: (L,C,H,W)
            # MSE over L steps
            # CELoss over L steps
            # h, w = self.perfect['image_cls'].shape[2:]
            # ground_mask = torch.tensor(layer0_mask(h, w))
            # perfect_obs = self.perfect['image_cls'] * ground_mask
            # realYolo_obs = self.realYolo['image_cls'] * ground_mask
            # whiteYolo_obs = self.whiteYolo['image_cls'] * ground_mask
            # perfect_obs = self.perfect['maps']
            # realYolo_obs = self.realYolo['maps']
            # whiteYolo_obs = self.whiteYolo['maps']
            perfect_obs = self.perfect['image_cls']
            realYolo_obs = self.realYolo['image_cls']
            whiteYolo_obs = self.whiteYolo['image_cls']

            perfect_p = self.perfect['delta']
            realYolo_p = self.realYolo['delta']
            whiteYolo_p = self.whiteYolo['delta']
        
            
            '''=======================Obs heatmap======================='''
            # print(f'perfect_obs:\n{perfect_obs}')
            # print(f'realYolo_obs:\n{realYolo_obs}')
            # print(f'whiteYolo_obs:\n{whiteYolo_obs}')
            p_obs = perfect_obs[:,1:].sum(1).numpy() #(L,H,W)
            r_obs = realYolo_obs[:,1:].sum(1).numpy() #(L,H,W)
            w_obs = whiteYolo_obs[:,1:].sum(1).numpy() #(L,H,W)
            heatmap_path = f'{out_path}/heatmap/{index}'
            try: os.makedirs(heatmap_path) 
            except: pass 
            for i in [0,1,2,3,5,10,30]:
                fig, axes = plt.subplots(2, 3, figsize=(10,6))
                plt.tick_params(labelsize=5)
                fig.suptitle(f"Yolo Output_step{i}")
                # fig.suptitle(f"Yolo InputObs_step{i}")
                
                sns_plot = sns.heatmap(p_obs[i], ax=axes[0,0], xticklabels=False, yticklabels=False)
                axes[0,0].set_title(f'PerfectYolo{perfect_p[i]}',fontsize=10)
                # axes[0,0].invert_yaxis()
                sns_plot = sns.heatmap(r_obs[i], ax=axes[0,1], xticklabels=False, yticklabels=False) 
                axes[0,1].set_title(f'OldPectYolo{realYolo_p[i]}',fontsize=10)
                # axes[0,1].invert_yaxis()
                sns_plot = sns.heatmap(w_obs[i], ax=axes[0,2], xticklabels=False, yticklabels=False)
                axes[0,2].set_title(f'SegmentYolo{whiteYolo_p[i]}',fontsize=10)
                # axes[0,2].invert_yaxis()
                sns_plot = sns.heatmap(perfect_obs[i,0], ax=axes[1,0], xticklabels=False, yticklabels=False)
                axes[1,0].set_title(f'PerfectYolo_layer0',fontsize=10)
                # axes[0,0].invert_yaxis()
                sns_plot = sns.heatmap(realYolo_obs[i,0], ax=axes[1,1], xticklabels=False, yticklabels=False) 
                axes[1,1].set_title(f'OldPectYolo_layer0',fontsize=10)
                # axes[0,1].invert_yaxis()
                sns_plot = sns.heatmap(whiteYolo_obs[i,0], ax=axes[1,2], xticklabels=False, yticklabels=False)
                axes[1,2].set_title(f'SegmentYolo_layer0',fontsize=10)
                # axes[0,2].invert_yaxis()
                fig.savefig(f'{heatmap_path}/step{i}.png', transparent=False, dpi=300, pad_inches = 0)
                plt.clf()
                plt.close()



            '''=======================Loss comapre=========================='''
            continue
            # CEloss = F.cross_entropy(perfect_obs, realYolo_obs,reduction='none')#(L,H,W)
            # Mseloss = F.mse_loss(perfect_obs, realYolo_obs,reduction='none') #(L,C,H,W)
            CEloss_realYolo = F.cross_entropy(realYolo_obs[:,1:], perfect_obs[:,1:],reduction='none') #(L,H,W)
            Mseloss_realYolo =  F.mse_loss(realYolo_obs[:,1:], perfect_obs[:,1:],reduction='none')  #(L,C,H,W)
            CEloss_whiteYolo = F.cross_entropy(whiteYolo_obs[:,1:], perfect_obs[:,1:],reduction='none') #(L,H,W)
            Mseloss_whiteYolo =  F.mse_loss(whiteYolo_obs[:,1:], perfect_obs[:,1:],reduction='none')  #(L,C,H,W)
            CEloss_real2white = F.cross_entropy(realYolo_obs[:,1:], whiteYolo_obs[:,1:],reduction='none') #(L,H,W)
            Mseloss_real2white =  F.mse_loss(realYolo_obs[:,1:], whiteYolo_obs[:,1:],reduction='none')  #(L,C,H,W)

            # print(f'CEloss:{CEloss.sum((1,2))}\nCEloss_item:{CEloss_realYolo.sum((1,2))}\nMseloss:{Mseloss.sum((1,2,3))}\nMseloss_item:{Mseloss_realYolo.sum((1,2,3))}')   
            print(f'CEloss_realYolo:{CEloss_realYolo.sum((1,2))}\nCEloss_whiteYolo:{CEloss_whiteYolo.sum((1,2))}\nCEloss_real2white:{CEloss_real2white.sum((1,2))}\n'
                  f'Mseloss_realYolo:{Mseloss_realYolo.sum((1,2,3))}\nMseloss_whiteYolo:{Mseloss_whiteYolo.sum((1,2,3))}\nMseloss_real2white:{Mseloss_real2white.sum((1,2,3))}')
            # draw loss over time
            fig_path=f'{out_path}/{index}.png'
            # plt.plot(np.arange(1,self.num_steps+1), CEloss.numpy().sum((1,2)) ,label='CEloss')
            plt.plot(np.arange(1,self.num_steps+1), CEloss_realYolo.numpy().sum((1,2)) ,label='CEloss_realYolo')
            plt.plot(np.arange(1,self.num_steps+1), CEloss_whiteYolo.numpy().sum((1,2)),label='CEloss_whiteYolo')
            # plt.plot(np.arange(1,self.num_steps+1), Mseloss.numpy().sum((1,2,3)),label='Mseloss')
            plt.plot(np.arange(1,self.num_steps+1), Mseloss_realYolo.numpy().sum((1,2,3)),label='Mseloss_realYolo')
            plt.plot(np.arange(1,self.num_steps+1), Mseloss_whiteYolo.numpy().sum((1,2,3)),label='Mseloss_whiteYolo')
            
            plt.legend()
            plt.savefig(fig_path)
            plt.figure().clear()

            
            heatmap_path = f'{out_path}/heatmap/{index}'
            try: os.makedirs(heatmap_path) 
            except: pass           
            
            for i in [0,10,30]:
                fig, axes = plt.subplots(2, 3)
                plt.tick_params(labelsize=5)
                fig.suptitle(f"Yolo comparison_step{i}")
                
                sns_plot = sns.heatmap(CEloss_realYolo[i], ax=axes[0,0], xticklabels=False, yticklabels=False,vmin=0,vmax=2)
                axes[0,0].set_title(f'CEloss_realYolo',fontsize=10)
                axes[0,0].invert_yaxis()
                sns_plot = sns.heatmap(CEloss_whiteYolo[i], ax=axes[0,1], xticklabels=False, yticklabels=False,vmin=0,vmax=2) 
                axes[0,1].set_title(f'CEloss_whiteYolo',fontsize=10)
                axes[0,1].invert_yaxis()
                sns_plot = sns.heatmap(CEloss_real2white[i], ax=axes[0,2], xticklabels=False, yticklabels=False,vmin=0,vmax=2)
                axes[0,2].set_title(f'CEloss_real2white',fontsize=10)
                axes[0,2].invert_yaxis()
                sns_plot = sns.heatmap(Mseloss_realYolo[i].numpy().sum(0), ax=axes[1,0], xticklabels=False, yticklabels=False,vmin=0,vmax=1)
                axes[1,0].set_title(f'Mseloss_realYolo',fontsize=10)
                axes[1,0].invert_yaxis()
                sns_plot = sns.heatmap(Mseloss_whiteYolo[i].numpy().sum(0), ax=axes[1,1], xticklabels=False, yticklabels=False,vmin=0,vmax=1)
                axes[1,1].set_title(f'Mseloss_whiteYolo',fontsize=10)
                axes[1,1].invert_yaxis()
                sns_plot = sns.heatmap(Mseloss_real2white[i].numpy().sum(0), ax=axes[1,2], xticklabels=False, yticklabels=False,vmin=0,vmax=1)
                axes[1,2].set_title(f'Mseloss_real2white',fontsize=10)
                axes[1,2].invert_yaxis()
                fig.savefig(f'{heatmap_path}/step{i}.png', transparent=False, dpi=300, pad_inches = 0)
                plt.clf()
                plt.close()

def layer0_mask(h,w):
    mask = np.zeros((h,w))
    row, col = h // 2, w // 2
    j = 0
    for i in range(row, h):
        l, r = col-j, col+j+1
        mask[i,l:r] = 1
        j += 1
    return mask

if __name__ == '__main__':
    a = preprocess()
    a.generate()
    # layer0_mask(11,11)
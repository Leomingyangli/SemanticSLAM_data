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
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='May22th_perfect_realYolo_UfUnkonwnYolo')  
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
    args.path_perfect = 'Jun6th_YoloSegment/Apr15th_perfect_m33obj21len31angl45_30env/semantic/val'
    args.path_realYolo = 'Jun6th_YoloSegment/Apr15th_realYolo_prj1_m33obj21len31angl45_30env/semantic/val'
    args.path_Yolosegment = 'Jun6th_YoloSegment/Apr15th_realYolo_UfUnknown_ceil01_m33obj21len31angl45_30env/semantic/val'
    # args.path_perfect = 'Apr15th_realYolo/Apr15th_perfect_m33obj21len31angl45_30env/semantic/val'
    # # args.path_realYolo = 'Gazebo_Apr15th_realYolo_scale3_2of30env_map33_obj21_len40_wo_rgbd_test_small.npz'
    # args.path_realYolo = 'Apr15th_realYolo/Apr15th_realYolo_prj1_m33obj21len31angl45_30env/semantic/val'
    # args.path_Yolosegment = 'Apr15th_realYolo/Apr15th_realYolo_UfUnknown_ceil01_m33obj21len31angl45_30env/semantic/val'
    # args.path_Yolosegment = 'May6th_YoloSegment/May6th_YoloSegment_ceilfloor01_m33obj21len31angl45_30env/semantic/val'
    # args.path_Yolosegment = 'Apr30th_YoloSegment/Apr30th_YoloSegment_m33obj21len31angl45_30env/semantic/val'
    # args.path_Yolosegment = 'Apr30th_YoloSegment/Apr30th_YoloSegment_prj1_m33obj21len31angl45_30env/semantic/val'
    # args.path_Yolosegment = 'Apr30th_YoloSegment/Apr30th_YoloSegment_prj01New_m33obj21len31angl45_30env/semantic/val'
    # args.path_Yolosegment = 'Apr30th_YoloSegment/Apr30th_YoloSegment_ceil01_m33obj21len31angl45_30env/semantic/val'
    # args.path_Yolosegment = 'Apr30th_YoloSegment/Apr30th_YoloSegment_ceilfloor005_m33obj21len31angl45_30env/semantic/val'
    # args.path_Yolosegment = 'Apr30th_YoloSegment/Apr30th_YoloSegment_ceilfloor001_m33obj21len31angl45_30env/semantic/val'
    
    
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

        out_path = f'printlog/yoloCompareTest/{self.args.date}'
        try:
            os.makedirs(out_path)
        except OSError:
            pass
        print('\n\nGenerate training file')

        self.nsplit = len(os.listdir(self.args.path_perfect)) # get total batch
        self.max_steps = 31
        print(f'number of files:{self.nsplit}')
        print(f'number of steps in one trajectory:{self.max_steps}')

        plt.ylabel('Error')
        plt.xlabel('Time steps')
        plt.grid(True,'major','y')
        for index in tqdm(range(self.nsplit)):
            print('='*20,'batch:', index, '='*20)
            data_perfect = np.load(os.path.join(self.args.path_perfect,str(index)+'.npz'),allow_pickle=True)
            data_realYolo = np.load(os.path.join(self.args.path_realYolo,str(index)+'.npz'),allow_pickle=True)
            data_YoloSeg = np.load(os.path.join(self.args.path_Yolosegment,str(index)+'.npz'),allow_pickle=True)
            self.obs_keys = list(data_perfect.files)
            self.perfect = {}
            self.realYolo = {}
            self.YoloSeg = {}
            # max_batch = data_file[self.obs_keys[0]].shape[1]
            for key_ in self.obs_keys:
                print(f'{key_}:{data_perfect[key_].shape}')
                if key_ in ['rgb','depth']: continue
                if 'labl' not in key_:
                    self.perfect[key_] = torch.tensor(data_perfect[key_][:self.num_steps],device=self.device).float()
                    self.realYolo[key_] = torch.tensor(data_realYolo[key_][:self.num_steps],device=self.device).float()
                    self.YoloSeg[key_] = torch.tensor(data_YoloSeg[key_][:self.num_steps],device=self.device).float()
                else:
                    self.perfect[key_] = torch.tensor(data_perfect[key_][index],device=self.device).float()
                    self.realYolo[key_] = torch.tensor(data_realYolo[key_][index],device=self.device).float()         
                    self.YoloSeg[key_] = torch.tensor(data_YoloSeg[key_][index],device=self.device).float()
            # observation size: (L,C,H,W)
            # MSE over L steps
            # CELoss over L steps
            # h, w = self.perfect['image_cls'].shape[2:]
            # ground_mask = torch.tensor(layer0_mask(h, w))
            # perfect_obs = self.perfect['image_cls'] * ground_mask
            # realYolo_obs = self.realYolo['image_cls'] * ground_mask
            # YoloSeg_obs = self.YoloSeg['image_cls'] * ground_mask
            # perfect_obs = self.perfect['maps']
            # realYolo_obs = self.realYolo['maps']
            # YoloSeg_obs = self.YoloSeg['maps']
            perfect_obs = self.perfect['image_cls']
            realYolo_obs = self.realYolo['image_cls']
            YoloSeg_obs = self.YoloSeg['image_cls']

            perfect_p = self.perfect['pose_changes']
            realYolo_p = self.realYolo['pose_changes']
            YoloSeg_p = self.YoloSeg['pose_changes']

            perfect_map = self.perfect['maps']
            realYolo_map = self.realYolo['maps']
            YoloSeg_map = self.YoloSeg['maps']
        
            
            '''=======================Obs heatmap======================='''
            # print(f'perfect_obs:\n{perfect_obs}')
            # print(f'realYolo_obs:\n{realYolo_obs}')
            # print(f'YoloSeg_obs:\n{YoloSeg_obs}')
            p_obs = perfect_obs[:,1:].sum(1).numpy() #(L,H,W)
            r_obs = realYolo_obs[:,1:].sum(1).numpy() #(L,H,W)
            ys_obs = YoloSeg_obs[:,1:].sum(1).numpy() #(L,H,W)
            
            p_map = perfect_map[:,1:].sum(1).numpy() #(L,H,W)
            r_map = realYolo_map[:,1:].sum(1).numpy() #(L,H,W)
            ys_map = YoloSeg_map[:,1:].sum(1).numpy() #(L,H,W)
            obs_r2p = r_obs - p_obs
            obs_ys2p = ys_obs - p_obs
            obs_r2ys = r_obs - ys_obs
            
            map_r2p = r_map - p_map
            map_ys2p = ys_map - p_map
            map_r2ys = r_map - ys_map
            for i in range(ys_obs.shape[0]):
                print(f'-------------step:{i}---------------')
                print(ys_obs[i])
                print()

            heatmap_path = f'{out_path}/heatmap/{index}'
            try: os.makedirs(heatmap_path) 
            except: pass 
            for i in [0,1,2,3,5,10,30]:
                fig, axes = plt.subplots(2, 3, figsize=(10,6))
                plt.tick_params(labelsize=5)
                fig.suptitle(f"Yolo Output_step{i}")
                # fig.suptitle(f"Yolo InputObs_step{i}")
                
                sns_plot = sns.heatmap(p_obs[i], ax=axes[0,0], xticklabels=False, yticklabels=False)
                axes[0,0].set_title(f'obs_Perfect',fontsize=10)
                # axes[0,0].invert_yaxis()
                sns_plot = sns.heatmap(r_obs[i], ax=axes[0,1], xticklabels=False, yticklabels=False) 
                axes[0,1].set_title(f'obs_RealYolo',fontsize=10)
                # axes[0,1].invert_yaxis()
                sns_plot = sns.heatmap(ys_obs[i], ax=axes[0,2], xticklabels=False, yticklabels=False)
                axes[0,2].set_title(f'obs_YoloUF',fontsize=10)
                # axes[0,2].invert_yaxis()
                sns_plot = sns.heatmap(p_map[i], ax=axes[1,0], xticklabels=False, yticklabels=False)
                axes[1,0].set_title(f'map_Perfect',fontsize=10)
                # axes[0,0].invert_yaxis()
                sns_plot = sns.heatmap(r_map[i], ax=axes[1,1], xticklabels=False, yticklabels=False) 
                axes[1,1].set_title(f'map_RealYolo',fontsize=10)
                # axes[0,1].invert_yaxis()
                sns_plot = sns.heatmap(ys_map[i], ax=axes[1,2], xticklabels=False, yticklabels=False)
                axes[1,2].set_title(f'map_YoloUF',fontsize=10)
                # sns_plot = sns.heatmap(obs_r2p[i], ax=axes[0,0], xticklabels=False, yticklabels=False)
                # axes[0,0].set_title(f'obs_RealYolo-Perfect',fontsize=10)
                # # axes[0,0].invert_yaxis()
                # sns_plot = sns.heatmap(obs_ys2p[i], ax=axes[0,1], xticklabels=False, yticklabels=False) 
                # axes[0,1].set_title(f'obs_YoloSeg-Perfect',fontsize=10)
                # # axes[0,1].invert_yaxis()
                # sns_plot = sns.heatmap(obs_r2ys[i], ax=axes[0,2], xticklabels=False, yticklabels=False)
                # axes[0,2].set_title(f'obs_RealYolo-YoloSeg',fontsize=10)
                # # axes[0,2].invert_yaxis()
                # sns_plot = sns.heatmap(map_r2p[i], ax=axes[1,0], xticklabels=False, yticklabels=False)
                # axes[1,0].set_title(f'map_RealYolo-Perfect',fontsize=10)
                # # axes[0,0].invert_yaxis()
                # sns_plot = sns.heatmap(map_ys2p[i], ax=axes[1,1], xticklabels=False, yticklabels=False) 
                # axes[1,1].set_title(f'map_YoloSeg-Perfect',fontsize=10)
                # # axes[0,1].invert_yaxis()
                # sns_plot = sns.heatmap(map_r2ys[i], ax=axes[1,2], xticklabels=False, yticklabels=False)
                # axes[1,2].set_title(f'map_RealYolo-YoloSeg',fontsize=10)
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
            CEloss_YoloSeg = F.cross_entropy(YoloSeg_obs[:,1:], perfect_obs[:,1:],reduction='none') #(L,H,W)
            Mseloss_YoloSeg =  F.mse_loss(YoloSeg_obs[:,1:], perfect_obs[:,1:],reduction='none')  #(L,C,H,W)
            CEloss_real2white = F.cross_entropy(realYolo_obs[:,1:], YoloSeg_obs[:,1:],reduction='none') #(L,H,W)
            Mseloss_real2white =  F.mse_loss(realYolo_obs[:,1:], YoloSeg_obs[:,1:],reduction='none')  #(L,C,H,W)

            # print(f'CEloss:{CEloss.sum((1,2))}\nCEloss_item:{CEloss_realYolo.sum((1,2))}\nMseloss:{Mseloss.sum((1,2,3))}\nMseloss_item:{Mseloss_realYolo.sum((1,2,3))}')   
            print(f'CEloss_realYolo:{CEloss_realYolo.sum((1,2))}\nCEloss_YoloSeg:{CEloss_YoloSeg.sum((1,2))}\nCEloss_real2white:{CEloss_real2white.sum((1,2))}\n'
                  f'Mseloss_realYolo:{Mseloss_realYolo.sum((1,2,3))}\nMseloss_YoloSeg:{Mseloss_YoloSeg.sum((1,2,3))}\nMseloss_real2white:{Mseloss_real2white.sum((1,2,3))}')
            # draw loss over time
            fig_path=f'{out_path}/{index}.png'
            # plt.plot(np.arange(1,self.num_steps+1), CEloss.numpy().sum((1,2)) ,label='CEloss')
            plt.plot(np.arange(1,self.num_steps+1), CEloss_realYolo.numpy().sum((1,2)) ,label='CEloss_realYolo')
            plt.plot(np.arange(1,self.num_steps+1), CEloss_YoloSeg.numpy().sum((1,2)),label='CEloss_YoloSeg')
            # plt.plot(np.arange(1,self.num_steps+1), Mseloss.numpy().sum((1,2,3)),label='Mseloss')
            plt.plot(np.arange(1,self.num_steps+1), Mseloss_realYolo.numpy().sum((1,2,3)),label='Mseloss_realYolo')
            plt.plot(np.arange(1,self.num_steps+1), Mseloss_YoloSeg.numpy().sum((1,2,3)),label='Mseloss_YoloSeg')
            
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
                sns_plot = sns.heatmap(CEloss_YoloSeg[i], ax=axes[0,1], xticklabels=False, yticklabels=False,vmin=0,vmax=2) 
                axes[0,1].set_title(f'CEloss_YoloSeg',fontsize=10)
                axes[0,1].invert_yaxis()
                sns_plot = sns.heatmap(CEloss_real2white[i], ax=axes[0,2], xticklabels=False, yticklabels=False,vmin=0,vmax=2)
                axes[0,2].set_title(f'CEloss_real2white',fontsize=10)
                axes[0,2].invert_yaxis()
                sns_plot = sns.heatmap(Mseloss_realYolo[i].numpy().sum(0), ax=axes[1,0], xticklabels=False, yticklabels=False,vmin=0,vmax=1)
                axes[1,0].set_title(f'Mseloss_realYolo',fontsize=10)
                axes[1,0].invert_yaxis()
                sns_plot = sns.heatmap(Mseloss_YoloSeg[i].numpy().sum(0), ax=axes[1,1], xticklabels=False, yticklabels=False,vmin=0,vmax=1)
                axes[1,1].set_title(f'Mseloss_YoloSeg',fontsize=10)
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
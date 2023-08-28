import os,sys,copy,glob,time,math,torch,cv2,scipy.stats
import numpy as np
import matplotlib.pyplot as plt
# import open3d as o3d
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
'''
Gnereate data with different environment(map)
first_release:
    Sep26th 2022
Each file slice the former 100 waypoints
Output:
    image_cls:      L,b,cls,obs_size,obs_size
    maps:           L,b,cls,map_size,map_size   
                    where cls = 1unknown+1ground+10obsjects
    delta:          L,b,3 (x,y,yaw) 
                    where yaw is in radian degree from 0 to 2pi
    
    map_labl:       map_size,map_size
    map_cls_labl    cls,map_size,map_size -convert map_labl to one hot tensor
output:
    image_cls_perf
    maps_perf
    imagle_cls_obst
    maps_obst
'''
torch.set_printoptions(linewidth=4000,precision=2,threshold=1000000,sci_mode=False)
np.set_printoptions(linewidth=10000,threshold=100000,precision=2,suppress=True)
np.random.seed(0)

def get_args():
    new_directory = "/home/mli170/SLAM_PROJECT/SemanticSLAM_data"  # Replace this with the desired directory path
    os.chdir(new_directory)

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='Aug20th_ceil1floor02_resnet_scale3')   
    # parser.add_argument('--obs_case', type=str, default='perfect',help='perfect/perfect_obstructed/real')
    parser.add_argument('--prj_up', type=bool, default=True)
    parser.add_argument('--prj_down', type=bool, default=True)
    parser.add_argument('--prj_threshold', type=float, default=0.02)
    parser.add_argument('--obs_softmax', type=str, default=False,help='apply softmax on ground input observation')
    parser.add_argument('--temperature', type=float, default=15,help='temperature of softmax')

    parser.add_argument('--map_size', type=int, default=11,help='The size of environment')
    parser.add_argument('--map_scale', type=int, default=3,help='The scale of unit, 1meter can be devided by 3 units')
    parser.add_argument('--obs_size', type=int, default=11,help='The size of environment')
    # parser.add_argument('--obs_threshold', type=float, default=0.5,help='threshold for valid observation')
    parser.add_argument('--n_object', type=int, default=40,help='Item quantities')
    parser.add_argument('--feature_dim', type=int, default=11, help='Item varieties + 1 ground per environment')
    parser.add_argument('--env_ctr', type=int, default=30, help='total environemtns')
    parser.add_argument('--tra_ctr', type=int, default=3, help='number of trajecotry per environment')
    parser.add_argument('--tra_len', type=int, default=80, help='number of steps per trajectory')
    parser.add_argument('--train_env', type=int, default=26, help='environment to train')
    parser.add_argument('--same_env', type=int, default=30, help='environment to train')
    parser.add_argument('--rot_mode', type=str, default='bilinear', help='bilinear or nearest')   

    parser.add_argument('--equal_invisible', type=bool, default=False)
    args = parser.parse_args()
    args.map_size *= args.map_scale
    args.obs_size *= args.map_scale
    
    args.out_path = f'data_npz'
    args.input_path = 'data_raw/2023_Jun19th_dataset_w_rgbd_raw_scale3_40objects'
    args.input_folder = 'data_3_seg_resnet'

    print('''data_3_seg_resnet has 11layers. in layer[0] 1 is ground, all 0 is invisible area, and each entry of layer is percentage of pixels for that label 
            Also, it contains low_level features from reset [position, ground_obs, ground_obs_semantic, ground_low_feature(1,64,h,w)]''')
    today = date.today()
    print(today.strftime("%b-%d-%Y"))
    print(sys.argv)
    pp.pprint(vars(args))
    return args


class Parameters():
    def __init__(self,args):
        self.device = torch.device('cpu')
        #Slam_System
        self.map_id = '1'
        self.previous_pose = [10,10]
        self.candidate = 1
        self.init_step = 100
        self.grid_unit = 1
        self.grid_count = int(1 /self.grid_unit)
        self.feature_unit = args.feature_dim
        #Map_System
        self.map_border = 2*2*self.grid_count
        self.map_size = args.map_size
        self.half_map_size = int(self.map_size/2)
        self.map_size = self.map_size + self.map_border
        self.map_size = self.map_size if self.map_size %2 ==1 else self.map_size+1

        self.local_map_size = args.obs_size
        self.local_map_size = self.local_map_size if self.local_map_size % 2 ==1 else self.local_map_size+1
        self.half_local_map_size = int(self.local_map_size/2)

        self.feat_dim_range = (0,self.feature_unit*self.grid_count)
        self.feat_dim = self.feat_dim_range[1] - self.feat_dim_range[0]
        self.map_scale = 1
        self.map_shape = (self.feat_dim, self.map_size, self.map_size)
        self.local_map_shape = (self.feat_dim, self.local_map_size, self.local_map_size)
        # print(f'map_shape:{self.map_shape}')
        # print(f'local_map_shape:{self.local_map_shape}')
        
        self.nangles = 36
        self.angles = torch.Tensor(np.radians(np.linspace(0, 360, self.nangles + 1)[:-1]-180)).to(self.device)
        self.zero_angle_idx = self.nangles //2

        self.update_rate = 0.26
        self.threshold = 0.1

        #self.label_map_path = '../data/4.6/env_31_30_300.npy'
        #self.train_data_path = '../data/4.6/path_31_30_300.npy'

        self.dataset = '../data/5.1/world_1.npy'

        self.label_map_path = '/data/mli170/2022_May30_partial_dataset/dataset'
        self.train_data_path = '/data/mli170/2022_May30_partial_dataset/dataset'

        #self.label_map_path = '../data/4.11/10_slam_label.npy'
        #self.train_data_path = '../data/4.11/10_slam_position_list.npy'

    def view_rgb_image(self,image):
        image = np.array(image)
        image = image.astype(np.uint8)
        #cv2.namedWindow("rgb_image",cv2.WINDOW_KEEPRATIO)
        #cv2.resizeWindow("rgb_image", 500, 500)
        cv2.imwrite('image.png',image)
        #cv2.imshow('rgb_image',image)
        #cv2.waitKey()

    def plot_3D(self,points,image=None):
        open3d=True
        if open3d:
            point=[]
            color=[]
            for i in range(len(points[0])):
                if image[0][i] == 0 and image[1][i] == 0 and image[2][i] == 0:
                    continue
                point.append([points[0][i]*self.grid_unit,points[1][i]*self.grid_unit,points[2][i]*self.grid_unit])
                color.append([image[2][i],image[1][i],image[0][i]])
            map = o3d.geometry.PointCloud()
            map.points = o3d.utility.Vector3dVector(point)
            if type(image) != type(None):
                map.colors = o3d.utility.Vector3dVector(color)
            cube_points = np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[1, 1, 0],[0, 0, 1],[1, 0, 1],[0, 1, 1],[1, 1, 1]]) * self.grid_count
            cube_lines = [[0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],[5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7]]
            cube_colors = [[0, 0, 0] for i in range(len(cube_lines))]
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(cube_points),lines=o3d.utility.Vector2iVector(cube_lines))
            line_set.colors = o3d.utility.Vector3dVector(cube_colors)
            o3d.visualization.draw_geometries([map,line_set])
        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_zlabel('Z', color='k')
            ax.set_ylabel('Y', color='k')
            ax.set_xlabel('X', color='k')
            if type(image) == type(None):
                ax.plot(points[0], points[1], points[2], marker = 'o')
            else:
                for i in range(len(points[0])):
                    if image[0][i] == 0 and image[1][i] == 0 and image[2][i] == 0:
                        continue
                    ax.plot([points[0][i]], [points[1][i]], [points[2][i]], marker = 'o' , color=(image[2][i],image[1][i],image[0][i]))
            ax.view_init(40, -130)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            plt.show()
        input()

class Observation_System():
    def __init__(self,p,args):
        self.p = p#Parameters()
        self.args = args
        # self.label_map = np.load(self.p.label_map_path,allow_pickle=True)
        # self.data = np.load(self.p.dataset,allow_pickle=True)
        self.grid_unit = self.p.grid_unit

    def linear_generate(self):
        # get observation from pose
        def center_observation(observation,center,x,y):
            # print(f'ori:\n{observation}')
            left,right,up,down = center[1], len(observation[0])-center[1]-1, center[0], len(observation)-center[0]-1
            observation = np.hstack((np.zeros((len(observation),y-left)),observation)) if left<y else observation if left==y else observation[:,(left-y):]
            observation = np.hstack((observation,np.zeros((len(observation),y-right)))) if right < y else observation  if right == y else observation[:,:(y-right)]
            observation = np.vstack((np.zeros((x-up,len(observation[0]))),observation)) if up < x else observation if up ==x else observation[up-x:,:]
            observation = observation = np.vstack((observation,np.zeros((x-down,len(observation[0]))))) if down < x else observation if down==x else observation[:(x-down),:]
            # print(f'observation:\n{observation}')
            return observation

        def semantic_tf(observation):
            observation_semantic = np.zeros((self.p.feat_dim_range[1],len(observation),len(observation[0])))
            for i in range(len(observation_semantic)):
                id = np.argwhere((observation == i))
                for j in id:
                    observation_semantic[i][j[0],j[1]]=1
                # sum = np.sum(observation_semantic[i])
                # if sum != 0:
                #     observation_semantic[i] /= sum
            # print(f'observation_semantic:{observation_semantic.shape}\n{observation_semantic}')
            return observation_semantic[self.p.feat_dim_range[0]:self.p.feat_dim_range[1]]

        def cut_observation(observation,msk_val=0):
            mask=np.zeros((len(observation),len(observation[0])))
            center = [int(len(observation)/2),int(len(observation[0])/2)]
            for i in range(center[0]+2):
                for j in range(i):
                    mask[center[0]-1+i][center[1]-j] = 1
                    mask[center[0]-1+i][center[1]+j] = 1
            # print(f'2nd mask:\n{mask}')
            # mask=np.zeros((len(observation),len(observation[0])))
            # center = [int(len(observation)/2),int(len(observation[0])/2)]
            # for i in range(center[0]+2):
            #     for j in range(i):
            #         mask[center[0]+1-i][center[1]-j] = 1
            #         mask[center[0]+1-i][center[1]+j] = 1
            # print(f'1st mask:\n{mask}')

            for i in range(int(len(observation))):
                for j in range(int(len(observation[0]))):
                    if(mask[i][j]==0):
                        observation[i][j]=msk_val
            # print(f'cut observation:\n{observation}')
            return observation

        def rotate_3D(observation,rad):
            for i in range(len(observation)):
                # print(f'bef_rotate:{observation[i]}')
                # observation_temp = rotate(observation[i], angle=degree,order=0,mode='constant')
                '''New rotate function!!!'''
                observation_temp = rotate_tensor(observation[i],rad,self.args.rot_mode)
                index_new = [int(len(observation_temp)/2),int(len(observation_temp[0])/2)]
                # print(f'bef_center:{observation_temp}')
                observation_temp = center_observation(observation_temp,index_new,self.p.half_local_map_size,self.p.half_local_map_size)
                # print(f'bef_cut:{observation_temp}')

                # mask value 0 at invisible areas in each layer
                observation_temp = cut_observation(observation_temp,0)
                # print(f'aft_cut:{observation_temp}')
                observation[i] = observation_temp
            # print(f'rotate observation:\n{observation}')
            return observation

        def gen(env_start,env_end):
            x,y,z=[],[],[]
            image_file_all=[]
            depth_file_all = []
            pose_file_all=[]
            obs_file_all=[]
            obs_perf_file_all, obs_obst_file_all = [], []
            map_file_all=[]
            map_perf_file_all, map_obst_file_all = [], []
            obs_file_abst_all = []
            scene_all=[]
            scene_semantic_all=[]

            # a=[]
            #total batch = env*tra*step
            for j in tqdm(range(env_start,env_end+1)):
                
                pose_all,obs_abst_all,env_all,env_seman_all=[],[],[],[]
                obs_all, obs_perf_all, obs_obst_all = [], [], []
                item_obs_all = []
                image_all, depth_all = [], []

                #get pose of one env
                env_map = copy.deepcopy(np.load(f'{self.args.input_path}/slam_{j}/label.npy',allow_pickle=True)) #h,w
                env_map =  F.interpolate(torch.tensor(env_map)[None,None,],scale_factor=self.args.map_scale,mode='nearest')[0,0].numpy()
                # print(f'Read map information:\n{env_map}')

                for k in range(1, 1 + self.args.tra_ctr):
                    # rgbdata = np.load(f'{self.args.input_path}/slam_{j}/raw/{k}.npy',allow_pickle=True,encoding='bytes') #(L,3)
                    data = np.load(f'{self.args.input_path}/slam_{j}/{self.args.input_folder}/{k}.npy',allow_pickle=True) #(L,3)
                    # [step][pose,rgb,d]
                    # rgbd = np.load(f'{self.args.input_path}/slam_{j}/camera/{k}.npy',allow_pickle=True) #(L,3) RGB
                    images = []
                    depths = []
                    # print(rgbdata.shape)
                    # for jj in rgbdata:
                    #     image = jj[1].astype(np.int32)
                    #     pointcloud = jj[2].astype(np.float64).reshape(image.shape)
                    #     images.append(image[::4,::4,:])
                    #     depths.append(pointcloud[::4,::4,:])

                    # # for jj in range(1, self.args.tra_len+1):
                    # #     image = np.array(cv2.imread(f'{self.args.input_path}/slam_{j}/image/{k}/{jj}.png')) #(h,w,3) BGR
                    # #     images.append(image) #(L,h,w,3)
                    # #     depth = rgbd[jj-1][2].reshape(image.shape)
                    # #     depths.append(depth)
                    # image_all.append(np.array(images[:self.args.tra_len]))#(L,h,w,3)
                    # depth_all.append(np.array(depths[:self.args.tra_len]))

                    # get pose and observation
                    train_data = []
                    real_obs = []
                    # real_abs_obs = []

                    
                    for i in range(len(data)):
                        # print(f"data:\n{data[i][0]}\n{data[i][1]}\n{data[i][2]}")
                        train_data.append(np.array(data[i][0])) # L,3
                        if data[i][2].ndim == 4:
                            _obs = data[i][2][0] # f,H,W
                        elif data[i][2].ndim == 3:
                            _obs = data[i][2] # f,H,W
                        else:
                            print('error')


                        # print(f'origin obs:\n{_obs}')
                        '''Proejct 1 to the ground instead of the confidence value'''
                        if self.args.prj_up:
                            _obs[_obs > self.args.prj_threshold] = 1
                            # print(f'prj_up_sample:\n{_obs}')
                        if self.args.prj_down: # delete noise to some extent
                            _obs[_obs < self.args.prj_threshold] = 0
                            # print(f'prj_down_sample:\n{_obs}')
                        # print(f'after prj1_ceilfloor_{self.args.prj_threshold}:\n{np.sum(_obs, axis=0)}')

                        # #define invisbile features
                        unobserved_mask = (1 - (np.sum(_obs, axis=0) > 0)).astype(np.int16)[np.newaxis,...] # (1,H,W) value is either 0 or 1

                        if self.args.obs_softmax:
                            softmax_obs = F.softmax(torch.tensor(_obs)*self.args.temperature, dim=0).numpy()
                            # print(f'softmax_sample:\n{softmax_obs}')
                            _obs = unobserved_mask * _obs + (1 - unobserved_mask) * softmax_obs

                        if not self.args.equal_invisible: # add another layer for invisible features
                            # if _obs = 0 tensor, softmax(_obs) = [1/11] * 11
                            _obs = np.concatenate([unobserved_mask, _obs])# f+1, H, W
                        else:
                            unobserved_layer = np.ones((self.args.feature_dim, _obs.shape[1], _obs.shape[2]), dtype=np.float32) / self.args.feature_dim # (f,H,W)
                            _obs += unobserved_layer * unobserved_mask
                        # print(f'new observation: \n{_obs}')
                        # print(f'new observation_12layer_sum: \n{np.sum(_obs,axis=0)}')
                        '''Concatenate low-level feautre to ground obervation'''
                        ground_feature = data[i][3][0] #32,h,w
                        for i in range(len(ground_feature)):
                            tmp = rotate(ground_feature[i], angle=180,order=1,mode='constant')
                            ground_feature[i] = tmp
                        _obs = np.concatenate([_obs, ground_feature]) #f+1+32,h,w
                        # print(f'obs:{_obs}')
                        # return

                        real_obs.append(_obs) # L,f+1+32,H,W

                        # real_abs_obs.append(np.array(data[i][1])) # L,H,W
                        
                    #     pose = data[i][0]
                    #     pose[2] = np.rad2deg(pose[2])
                    #     print(f'pose:{data[i][0]}\nr_obs:\n{data[i][2][0]},\nabs_obs:\n{data[i][1]}')
                    #     print(f'env:{env_map.shape}\n{env_map}')
                    #     # print(np.array(data[i][2][0]).shape)
                    env_all.append(env_map) #B,H,W
                    env_semantic = np.zeros((self.args.feature_dim,env_map.shape[0],env_map.shape[1]))
                    for dd in range(env_semantic.shape[0]):
                        id = np.argwhere(env_map == dd)
                        for jf in id:
                            env_semantic[dd][jf[0],jf[1]]=1
                    env_seman_all.append(env_semantic) #B,f,H,W
                    obs_all.append(np.array(real_obs[:self.args.tra_len])) # B,L,f+1,H,W
                    # obs_abst_all.append(np.array(real_abs_obs[:self.args.tra_len]))
                    pose_all.append(np.array(train_data[:self.args.tra_len])) # B,L,3

                # quicj way- abandoned!!!!!
                # image_all = np.transpose(np.array(image_all),(1,0,4,2,3)) # B,L,h,w,3 -> L,B,3,h,w
                # depth_all = np.transpose(np.array(depth_all),(1,0,4,2,3))
                # pose_all = np.transpose(np.array(pose_all),(1,0,2)) # L,B,3
                # image_file_all        = image_all if len(image_file_all)==0 else np.concatenate((image_file_all,image_all),axis=1) #image L,B,3,H,W
                # depth_file_all        = depth_all if len(depth_file_all)==0 else np.concatenate((depth_file_all,depth_all),axis=1) #image L,B,3,H,W
                # pose_file_all         = pose_all if len(pose_file_all)==0 else np.concatenate((pose_file_all,pose_all),axis=1) #delta (L,b,3)
                # continue
                # generate perfect_observation and obstructe_observation
                for b, r_data in enumerate(pose_all):
                    obs_perf, obs_obst = [], []
                    for l, pose in enumerate(r_data):
                        x, y = int(pose[0]*self.args.map_scale), int(pose[1]*self.args.map_scale)
                        index = [x,y]
                        angle_rd = pose[2] # rotate clockwise to x-positive axis
                        observation_raw = copy.deepcopy(env_map)
                        observation = center_observation(observation_raw,index,self.p.half_local_map_size,self.p.half_local_map_size)
                        observation_semantic = semantic_tf(observation)
                        observation_semantic = rotate_3D(observation_semantic,angle_rd)

                        perfect_observation_semantic = observation_semantic #(f,h,w)

                        real_observation_semantic = obs_all[b][l] #(1+f,h,w)
                        
                        perfect_obstructed_observation_semantic = perfect_observation_semantic * real_observation_semantic[1:12].astype(bool)

                        # invisible layer
                        perfect_unobserved_mask = (1 - (np.sum(perfect_observation_semantic, axis=0) > 0)).astype(np.int16)[np.newaxis,...] # (1,H,W)
                        perfect_observation_semantic = np.concatenate([perfect_unobserved_mask, perfect_observation_semantic])# f+1, H, W
                        perfect_obstructed_unobserved_mask = (1 - (np.sum(perfect_obstructed_observation_semantic, axis=0) > 0)).astype(np.int16)[np.newaxis,...] # (1,H,W)
                        perfect_obstructed_observation_semantic = np.concatenate([perfect_obstructed_unobserved_mask, perfect_obstructed_observation_semantic])# f+1, H, W
                        
                        obs_perf.append(perfect_observation_semantic) #(L, f+1, H, W)
                        obs_obst.append(perfect_obstructed_observation_semantic) #(L, f+1, H, W)
                    obs_perf_all.append(np.array(obs_perf[:self.args.tra_len]))
                    obs_obst_all.append(np.array(obs_obst[:self.args.tra_len]))
                # image_all = np.transpose(np.array(image_all),(1,0,4,2,3)) # B,L,h,w,3 -> L,B,3,h,w
                # depth_all = np.transpose(np.array(depth_all),(1,0,4,2,3))
                obs_all = np.transpose(np.array(obs_all),(1,0,2,3,4)) # L,B,f,H,W
                obs_perf_all = np.transpose(np.array(obs_perf_all),(1,0,2,3,4)) # L,B,f,H,W
                obs_obst_all = np.transpose(np.array(obs_obst_all),(1,0,2,3,4)) # L,B,f,H,W
                pose_all = np.transpose(np.array(pose_all),(1,0,2)) # L,B,3

                # label ground layer with 1 in the visible area
                # obs_all[:,:,0] = 1 - obs_item_all

                '''Mask out invisible areas'''
                # ground_mask = layer0_mask(*obs_all.shape[3:]) 
                # obs_all = obs_all * ground_mask

                item_obs_all = np.sum(obs_all[:,:,1:], axis=2) # L,B,H,W
                item_obs_perf_all = np.sum(obs_perf_all[:,:,1:], axis=2) # L,B,H,W
                item_obs_obst_all = np.sum(obs_obst_all[:,:,1:], axis=2) # L,B,H,W
                # obs_abst_all = np.transpose(np.array(obs_abst_all),(1,0,2,3)) # L,B,H,W
                
                # maps_channel, maps_abstract = generate_maps(pose_all,np.array(env_all),obs_abst_all,self.args.feature_dim,self.args.map_scale) #L,B,f,H,W
                # maps_channel, maps_abstract = generate_maps_accurate(pose_all,np.array(env_all),obs_item_all,self.args.feature_dim,self.args.map_scale,equal_invisible=self.args.equal_invisible)
                
                # beacuse real_obs contains noise, which leadings to item_obs_all contain more visible area it shouldn`t be. This cause map_all contains same as map_perf_all.
                # To avoid above scenario, instead of using item_obs_all, we choose item_obs_obst_all, which filter out noise part.
                
                # map_all, _ = generate_maps_realPerfect(pose_all,np.array(env_seman_all),item_obs_all,self.args.feature_dim,self.args.map_scale,equal_invisible=self.args.equal_invisible)
                map_all, _ = generate_maps_realPerfect(pose_all,np.array(env_seman_all),item_obs_obst_all,self.args.feature_dim,self.args.map_scale,equal_invisible=self.args.equal_invisible)
                map_perf_all, _ = generate_maps_realPerfect(pose_all,np.array(env_seman_all),item_obs_perf_all,self.args.feature_dim,self.args.map_scale,equal_invisible=self.args.equal_invisible)
                map_obst_all, _ = generate_maps_realPerfect(pose_all,np.array(env_seman_all),item_obs_obst_all,self.args.feature_dim,self.args.map_scale,equal_invisible=self.args.equal_invisible)
                
                obs_file_all = obs_all  if len(obs_file_all)==0 else np.concatenate((obs_file_all,obs_all),axis=1) #img_cls (L,b,cls,H,W)
                obs_perf_file_all = obs_perf_all  if len(obs_perf_file_all)==0 else np.concatenate((obs_perf_file_all,obs_perf_all),axis=1) #img_cls (L,b,cls,H,W)
                obs_obst_file_all = obs_obst_all  if len(obs_obst_file_all)==0 else np.concatenate((obs_obst_file_all,obs_obst_all),axis=1) #img_cls (L,b,cls,H,W)
                image_file_all        = image_all if len(image_file_all)==0 else np.concatenate((image_file_all,image_all),axis=1) #image L,B,3,H,W
                depth_file_all        = depth_all if len(depth_file_all)==0 else np.concatenate((depth_file_all,depth_all),axis=1) #image L,B,3,H,W
                # obs_file_abst_all     = obs_abst_all if len(obs_file_abst_all)==0 else np.concatenate((obs_file_abst_all,obs_abst_all),axis=1) #img_cls (L,b,H,W)

                map_file_all = map_all if len(map_file_all)==0 else np.concatenate((map_file_all,map_all),axis=1) #maps: (L,b,f,h,w)
                map_perf_file_all = map_perf_all if len(map_perf_file_all)==0 else np.concatenate((map_perf_file_all,map_perf_all),axis=1) #maps: (L,b,f,h,w)
                map_obst_file_all = map_obst_all if len(map_obst_file_all)==0 else np.concatenate((map_obst_file_all,map_obst_all),axis=1) #maps: (L,b,f,h,w)
                
                pose_file_all         = pose_all if len(pose_file_all)==0 else np.concatenate((pose_file_all,pose_all),axis=1) #delta (L,b,3)
                scene_all     = np.array(env_all) if len(scene_all)==0 else np.concatenate((scene_all,np.array(env_all)),axis=0) #env information: (b,h,w)
                scene_semantic_all = np.array(env_seman_all) if len(scene_semantic_all)==0 else np.concatenate((scene_semantic_all,np.array(env_seman_all)),axis=0) #maps: (b,cls,h,w)
            print(f'pose_file_all:\n{pose_file_all.shape}')
            print(f'obs_file_all:\n{obs_file_all.shape}\nmap_file_all:\n{map_file_all.shape}')
            # print(f'obs_file_all obs_obst_file_all obs_perf_file_all:\n{obs_file_all.shape}{obs_obst_file_all.shape}{obs_perf_file_all.shape}\n')
            # print(f'map_file_all map_obst_file_all map_perf_file_all:\n{map_file_all.shape}{map_obst_file_all.shape}{map_perf_file_all.shape}\n')
            print(f'scene_all:\n{scene_all.shape}\nscene_semantic_all:\n{scene_semantic_all.shape}')
            for i in range(30):
                print(f'step:{i}')
                # print(f'pose_file_all:\n{pose_file_all[i,0,:2]}{np.rad2deg(pose_file_all[i,0,2])}')
                # print(f'perf_obs-obst_obs:\n{obs_perf_file_all[i,0,2:5]-obs_obst_file_all[i,0,2:5]}')
                # print(f'perf_obs-real_obs:\n{obs_perf_file_all[i,0,2:5]-obs_file_all[i,0,2:5]}')

                # print(f'perf_map-obst_map:\n{map_perf_file_all[i,0,2:5]-map_obst_file_all[i,0,2:5]}')
                # print(f'perf_map-real_map:\n{map_perf_file_all[i,0,2:5]-map_file_all[i,0,2:5]}')
                print(f'pose_file_all:\n{pose_file_all[i,0]}')
                print(f'obs_file_all:\n{obs_file_all[i,0,:3]}')
                # print(f'obs_obst_file_all:\n{obs_obst_file_all[i,0,:3]}')
                # print(f'obs_perf_file_all:\n{obs_perf_file_all[i,0,:3]}')
                print(f'map_file_all:\n{map_file_all[i,0,:3]}')
                # print(f'map_obst_file_all:\n{map_obst_file_all[i,0,:3]}')
                # print(f'map_perf_file_all:\n{map_perf_file_all[i,0,:3]}')
                print(f'scene_all:\n{scene_all[0]}')
                print(f'scene_semantic_all:\n{scene_semantic_all[0]}')
            
            # mean = []
            # std = []
            # for i in range(image_file_all.shape[1]):
            #     image_mean = torch.mean(torch.tensor(image_file_all[:,i]/255,device='cuda:0'), (0,2,3))
            #     mean.append(image_mean)
            #     image_std = torch.std(torch.tensor(image_file_all[:,i]/255,device='cuda:0'), (0,2,3))
            #     std.append(image_std)
            # mean = torch.mean(torch.stack(mean,dim=0),0)
            # std = torch.mean(torch.stack(std,dim=0),0)
            # print(f'image_mean:{mean}\t image_std:{std}')
            return image_file_all, depth_file_all, obs_file_all, pose_file_all, map_file_all, scene_all, scene_semantic_all
            # return image_file_all, depth_file_all, [obs_file_all, obs_obst_file_all, obs_perf_file_all], pose_file_all, [map_file_all,map_obst_file_all,map_perf_file_all], scene_all, scene_semantic_all
       
       #========================Training and testing on different env==========================
        print('='*20,'Training and testing on different env', '='*20,'\n')
        print('-'*20,'training', '-'*20)
        img, dep, obs, pos, maps, env_maps, env_smaps = gen(1,self.args.train_env)
        # print(f'img:{img.shape}, \ndep:{dep.shape}, \nobs:{obs.shape}, \npos:{pos.shape}, \nmaps:{maps.shape}, \nenv_maps:{env_maps.shape}, \nenv_smaps:{env_smaps.shape}')
        np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_CrossScene_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_train',
            rgb=img, depth=dep, image_cls=obs, delta=pos, maps=maps, map_labl=env_maps, map_cls_labl=env_smaps)
            # rgb=img, depth=dep, image_cls=obs[0], image_cls_obst=obs[1],image_cls_perf=obs[2], delta=pos, maps=maps[0],maps_obst=maps[1],maps_perf=maps[2], map_labl=env_maps, map_cls_labl=env_smaps)

        print('-'*20,'testing ', '-'*20,'\n')
        print('-'*20,f'Env index {self.args.train_env+1}-{self.args.env_ctr}', '-'*20,'\n')
        img, dep, obs, pos, maps, env_maps, env_smaps = gen(self.args.train_env+1,self.args.env_ctr)
        print('-'*20,'testing', '-'*20)
        remainder = self.args.env_ctr - self.args.train_env 
        # print(f'img:{img.shape}, \ndep:{dep.shape}, \nobs:{obs.shape}, \npos:{pos.shape}, \nmaps:{maps.shape}, \nenv_maps:{env_maps.shape}, \nenv_smaps:{env_smaps.shape}\n')
        np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_CrossScene_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_test',
            rgb=img, depth=dep, image_cls=obs, delta=pos, maps=maps, map_labl=env_maps, map_cls_labl=env_smaps)
            # rgb=img, depth=dep, image_cls=obs[0], image_cls_obst=obs[1],image_cls_perf=obs[2], delta=pos, maps=maps[0],maps_obst=maps[1],maps_perf=maps[2], map_labl=env_maps, map_cls_labl=env_smaps)

        #========================Training and testing on same env==========================
        print('\n','='*20,'Training and testing on same env but different trajecotry', '='*20)
        img, dep, obs, pos, maps, env_maps, env_smaps = gen(1,self.args.same_env)
        # # print(f'img:{img.shape}, \ndep:{dep.shape}, \nobs:{obs.shape}, \npos:{pos.shape}, \nmaps:{maps.shape}, \nenv_maps:{env_maps.shape}, \nenv_smaps:{env_smaps.shape}\n')
        B = pos.shape[1]
        sample_length = 3 # sample i for testing every 3 trajectories
        train_index, test_index = split_batch(B,sample_length) 

        # # np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_multiSessionEnv_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_all',
        #     # image_cls=obs, delta=pos, maps=maps, map_labl=env_maps, map_cls_labl=env_smaps)
        np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_IntraScene_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_train',
            rgb=img, depth=dep, 
            image_cls=obs[:,train_index,...], 
            # image_cls=obs[0][:,train_index,...],image_cls_obst=obs[1][:,train_index,...],image_cls_perf=obs[2][:,train_index,...], 
            delta=pos[:,train_index,...], 
            maps=maps[:,train_index,...], 
            # maps=maps[0][:,train_index,...],maps_obst=maps[1][:,train_index,...],maps_perf=maps[2][:,train_index,...], 
            map_labl=env_maps[train_index,...], map_cls_labl=env_smaps[train_index,...])
        np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_IntraScene_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_test',
            rgb=img, depth=dep, 
            image_cls=obs[:,test_index,...], 
            # image_cls=obs[0][:,test_index,...],image_cls_obst=obs[1][:,test_index,...],image_cls_perf=obs[2][:,test_index,...], 
            delta=pos[:,test_index,...], 
            maps=maps[:,test_index,...], 
            # maps=maps[0][:,test_index,...],maps_obst=maps[1][:,test_index,...],maps_perf=maps[2][:,test_index,...], 
            map_labl=env_maps[test_index,...], map_cls_labl=env_smaps[test_index,...])
        # print('\n','-'*30,'Sample', '-'*30)
        # print(f'pose:{pos[0,0]}\n\nobs:\n{obs[0,0]}\nmap:\n{maps[0,0]}\nenvironment:\n{env_maps[0]}')

    def image_generate(self,points,image=None):
        if type(image) != type(None):
            #self.p.view_rgb_image(image)
            image = image.reshape(-1,3)
            image = image/255
            image_t = np.transpose(image)
        points_t = np.transpose(points)
        points_t_min = np.ndarray.min(points_t,axis=1)
        x,y,z = points_t[2],points_t[0],points_t[1]
        z_min = np.ndarray.min(points_t,axis=1)
        z = z-np.ndarray.min(z)
        z = np.ndarray.max(z) - z
        #self.p.plot_3D([x,y,z],image_t)
        x = (x/self.grid_unit).astype(np.int32)
        y = (y/self.grid_unit).astype(np.int32)
        z = (z/self.grid_unit).astype(np.int32)
        #self.p.plot_3D([x,y,z],image_t)

        observation = np.zeros((self.p.feat_dim_range[1],self.p.local_map_size,self.p.local_map_size))
        observation_count = np.zeros((self.p.feat_dim_range[1],self.p.local_map_size,self.p.local_map_size))

        for i in range(len(x)):
            if x[i]<self.p.local_map_size and abs(y[i])<=self.p.half_local_map_size and z[i]<self.p.feat_dim_range[1]/self.p.feature_unit and z[i]>=0:
                y_tf = self.p.half_local_map_size - y[i]
                if type(image) == type(None):
                    observation[z[i],x[i],y_tf]+=1
                else:
                    observation[z[i]*3,x[i],y_tf] += image_t[0][i]
                    observation[z[i]*3+1,x[i],y_tf] += image_t[1][i]
                    observation[z[i]*3+2,x[i],y_tf] += image_t[2][i]
                    observation_count[z[i]*3,x[i],y_tf] += 1
                    observation_count[z[i]*3+1,x[i],y_tf] += 1
                    observation_count[z[i]*3+2,x[i],y_tf] += 1

        observation_new = []
        observation_count_new = []
        for i in range(self.p.feat_dim_range[1]):
            observation_new.append(np.vstack((np.zeros((self.p.half_local_map_size,len(observation[0]))),observation[i])))
            observation_count_new.append(np.vstack((np.zeros((self.p.half_local_map_size,len(observation_count[0]))),observation_count[i])))
        observation = np.array(observation_new)
        observation = observation[:,:-self.p.half_local_map_size]

        observation_count = np.array(observation_count_new)
        observation_count = observation_count[:,:-self.p.half_local_map_size]

        if type(image) == type(None):
            for i in range(self.p.feat_dim_range[1]):
                sum = np.sum(observation[i])
                if sum !=0:
                    observation[i]/=sum
        else:
            observation = np.divide(observation, observation_count, out=np.zeros_like(observation), where=observation_count!=0)

        index = np.argwhere(observation!=0)
        index_t = np.transpose(index)
        if type(image) != type(None):
            b,g,r=[],[],[]
            for i in index:
                if i[0]%3==0:
                    b.append(observation[i[0],i[1],i[2]])
                elif i[0]%3==1:
                    g.append(observation[i[0],i[1],i[2]])
                elif i[0]%3==2:
                    r.append(observation[i[0],i[1],i[2]])
            index = index[:int(len(index)/3)]
            index_t = np.transpose(index)
        #print(len(b))
        #print(len(g))
        #print(len(r))

        #self.p.plot_3D([index_t[1],index_t[2],index_t[0]],[b,g,r])
        observation = observation[self.p.feat_dim_range[0]:self.p.feat_dim_range[1]]
        observation = torch.tensor(observation,dtype=torch.float32,device=self.p.device)[None,:]
        return  self.rotation_resample(observation)

    def rotation_resample(self,x,offset=0.0):
        def rotate_tensor(r, t):
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
            A[:, 0, 0] = cos_t
            A[:, 0, 1] = sin_t
            A[:, 1, 0] = -sin_t
            A[:, 1, 1] = cos_t
            grid   = F.affine_grid(A, r.size(), align_corners=False)
            r_rot  = F.grid_sample(r, grid, mode=self.args.rot_mode, align_corners=False)


            return r_rot
        """
        Inputs:
            x       - (bs, f, s, s) feature maps
            angles_ - (nangles, ) set of angles to sample
        Outputs:
            x_rot   - (bs, nangles, f, s, s)
        """
        angles      = self.p.angles.clone() # (nangles, )
        bs, f, s, s = x.shape
        nangles     = angles.shape[0]
        x_rep       = x.unsqueeze(1).expand(-1, nangles, -1, -1, -1) # (bs, nangles, f, s, s)
        x_rep       = rearrange(x_rep, 'b o e h w -> (b o) e h w')
        angles      = angles.unsqueeze(0).expand(bs, -1).contiguous().view(-1) # (bs * nangles, )
        x_rot       = rotate_tensor(x_rep, angles + math.radians(offset)) # (bs * nangles, f, s, s)
        x_rot       = rearrange(x_rot, '(b o) e h w -> b o e h w', b=bs)

        return x_rot

def generate_maps(pose_all,label_map,obs_all,feature_dim,scale):
    '''
    Generate label map based on previous observation for each time step
    If the top left value of the grid is in FOV, the whole grid is regarded as in the FOV
    input: 
        pose_all: (L,B,3)
        label_map: (H,W)
        obs_all:(L,B,H,W)
        scale: int map scale
    output:
        maps_channel: (L,B,C,H,W) maps per step with item channel
        maps_abstract: (L,B,H,W) maps per step without item channel
    '''
    observation_raw = copy.deepcopy(label_map)  #H,W
    pose_copy = copy.deepcopy(pose_all)
    maps_channel = []
    maps_abstract = []
    semantic=True
    # print(f'ENV:\n{label_map}')
    #iterate over all poses
    for b in range(pose_copy.shape[1]):
        maps_abst = []
        maps_chnl = []
        map_traj = observation_raw[b]
        mask_raw = np.zeros_like(map_traj) #update mask for a trajecotry with L poses
        mask_all = np.zeros_like(map_traj) #update mask for a trajecotry with L poses
        # print(f'start')
        for l in range(pose_copy.shape[0]): 
            pose = pose_copy[l][b]
            pose[0] *= scale
            pose[1] *= scale
            img = obs_all[l][b]
            mask = coord_in_range(pose,copy.deepcopy(mask_raw))
            # print(f'pose:\n{pose[0],pose[1],math.degrees(pose[2])}')
            map_cur = mask*map_traj
            # print(f'pose:\n{pose_all[l][b]}')
            # print(f'bef maps:\n{maps}')
            # print(f'obs:\n{obs_all[l][b]}')

            # delete the border problem between observation and map

            # print(f'obs_all:\n{obs_all[l,b]}')
            for i in range(1,feature_dim):
                # if exist in img but no exist in obsMask -> use _linMask to add it to _map
                if np.argwhere(img == i).any() and not np.argwhere(map_cur == i).any():
                    candidates = np.argwhere(map_traj == i) # find all candidates
                    # print(f'Find {i}th item missing')

                    for x,y in candidates:
                        if mask_all[x,y]: # candidates in linMask
                            mask[x,y] = i
                # print(f'map={i}\n',np.argwhere(maps == i))
                # print(f'obs={i}\n',np.argwhere(obs_all[l][b] == i))
                if np.argwhere(map_cur == i).any() and not np.argwhere(img == i).any():
                    # print(f'Find {i}th item conflicts')
                    candidates = np.argwhere(map_cur == i)
                    for j in candidates:
                        mask[j[0],j[1]] = 0
            mask_all = np.logical_or(mask_all, mask)
            maps = mask_all * map_traj
                    # print(f'aft maps:\n{maps}')
            # print(f'mask:\n{mask}')
            map_semantic = np.zeros((feature_dim,mask.shape[0],mask.shape[1]))
            for i in range(map_semantic.shape[0]):
                id = np.argwhere(maps == i)
                for j in id:
                    map_semantic[i][j[0],j[1]]=1
            # print(f'map_semantic:\n{map_semantic}')
            maps_chnl.append(map_semantic) #Cls,H,W
            maps_abst.append(maps)# H,W
        maps_channel.append(np.array(maps_chnl)) 
        maps_abstract.append(np.array(maps_abst)) 
    maps_channel = np.transpose(np.array(maps_channel),(1,0,2,3,4)) #B,L,C,H,W -> L,B,C,H,W
    maps_abstract = np.transpose(np.array(maps_abstract),(1,0,2,3)) #B,L,H,W -> L,B,H,W
    # print(f'maps:\n{observation_raw}')
    # print(f'masks:\n{masks}')
    return (maps_channel,maps_abstract)

def generate_maps_accurate(pose_all,label_map,obs_all,feature_dim,scale,equal_invisible=False):
    '''
    Generate label map based on previous observation for each time step
    input: 
        pose_all: (L,B,3)
        label_map: (B,H,W)
        obs_all:(L,B,H,W)
        scale: int map scale
    output:
        maps_channel: (L,B,C,H,W) maps per step with item channel
        maps_abstract: (L,B,H,W) maps per step without item channel
    '''
    observation_raw = copy.deepcopy(label_map)  #B,H,W

    # get relative pose to transform image back
    pose_copy = torch.tensor(pose_all)
    pose_change = pose_copy.clone()
    L = pose_copy.shape[0]
    start_pose = pose_copy[0].clone()
    for t in range(L):
        pose_change[t] = compute_relative_pose_ori(start_pose, pose_copy[t])

    maps_channel = []
    maps_abstract = []
    #iterate over all poses
    for b in range(pose_copy.shape[1]):
        maps_abst = []
        maps_chnl = []
        # print(f'start')
        map_traj = observation_raw[b]

        emptyMask = np.zeros_like(map_traj)
        obsMask = np.zeros_like(map_traj) 
        linMask = np.zeros_like(map_traj)

        for l in range(pose_change.shape[0]): 
            r_pose = pose_change[l][b]
            r_pose[0] *= scale
            r_pose[1] *= scale
            pose = pose_copy[l][b].numpy()
            pose[0] *= scale
            pose[1] *= scale
            # print(f'==========================batch:{b},step:{l}==========================')
            # print(f'pose:\n{pose}')
            img = obs_all[l][b]            
            _obsMask = np.logical_or(affineGrid(img, r_pose), emptyMask)
            # print(f'obsMask_final:\n{_obsMask}')
            _linMask = coord_in_range(pose,copy.deepcopy(emptyMask))
            # print(f'linMask:\n{_linMask}')
            # _map = _obsMask * _linMask * map_traj           
            # for i in range(1,feature_dim):
            #     # if exist in img but no exist in obsMask -> use _linMask to add it to _map
            #     if np.argwhere(img == i).any() and not np.argwhere(_map == i).any():
            #         candidates = np.argwhere(map_traj == i) # find all candidates
            #         print(f'Find {i}th item missing')
            #         for x,y in candidates:
            #             if _linMask[x,y]: # candidates in linMask
            #                 _obsMask[x,y] = i
            
            #     # if exist in map but no exist in img -> delete
            #     if np.argwhere(_map == i).any() and not np.argwhere(img == i).any():
            #         candidates = np.argwhere(_map == i)
            #         print(f'Find {i}th item redundant')
            #         for x, y in candidates:
            #             _obsMask[x, y] = 0
            # print('after:',_obsMask.astype(np.int32))
            obsMask = np.logical_or(obsMask, _obsMask)
            linMask = np.logical_or(linMask, _linMask)
            map_updated = map_traj * obsMask * linMask
            # print(f'maps_update:\n{map_updated}')
            # print(f'env:\n{map_traj}')
                    
            map_semantic = np.zeros((feature_dim, label_map.shape[1], label_map.shape[2]))
            for i in range(map_semantic.shape[0]):
                id = np.argwhere(map_updated == i)
                for j in id:
                    map_semantic[i][j[0],j[1]]=1
            if equal_invisible:
                map_unknown = np.ones((feature_dim, label_map.shape[1], label_map.shape[2]),dtype=np.float32)/feature_dim
                map_semantic = linMask * map_semantic + (1 - linMask) * map_unknown
            # mask out invisible areas
            # map_semantic = map_semantic * linMask
            # map_semantic = map_semantic
            maps_chnl.append(map_semantic) #Cls,H,W
            maps_abst.append(map_updated)# H,W
        maps_channel.append(np.array(maps_chnl)) 
        maps_abstract.append(np.array(maps_abst)) 
    maps_channel = np.transpose(np.array(maps_channel),(1,0,2,3,4)) #B,L,C,H,W -> L,B,C,H,W
    maps_abstract = np.transpose(np.array(maps_abstract),(1,0,2,3)) #B,L,H,W -> L,B,H,W
    # print(f'maps:\n{observation_raw}')
    # print(f'masks:\n{masks}')
    return (maps_channel,maps_abstract)

def generate_maps_realPerfect(pose_all, semantic_env, obs_all, feature_dim, scale, equal_invisible=False):
    '''
    Generate label map based on intersection of perfect obs and real obs
    total layer = item layers + ground layer + unobserved layer
    input: 
        pose_all: (L,B,3)
        semantic_env: (B,f,H,W) Scene information without invisible
        item_obs_all:(L,B,h,w)
        scale: int map scale
    output:
        maps_channel: (L,B,C,H,W) maps per step with item channel
        maps_abstract: (L,B,H,W) maps per step without item channel
    '''
    H, W = semantic_env.shape[2], semantic_env.shape[3]
    pose_copy = torch.tensor(pose_all)

    # visible mask
    h, w =obs_all.shape[2], obs_all.shape[3]

    maps_channel = []
    maps_abstract = []
    #iterate over all poses
    for b in range(pose_copy.shape[1]):
        maps_abst = []
        maps_chnl = []
        # print(f'start')
        env_traj = semantic_env[b] #f,H,W

        obsMask = np.zeros((H,W))
        linMask = np.zeros((H,W))

        for l in range(pose_copy.shape[0]): 
            pose = pose_copy[l][b].numpy()
            pose[0] *= scale
            pose[1] *= scale

            # visible_mask = obs_all[l,b] # include ground and items
            # # _linMask = coord_in_range(pose,np.zeros((H,W))) # (h,w)
            # _linMask = mask_rotate(visible_mask, np.rad2deg(pose[2]), [int(pose[0]), int(pose[1])],h,w,h//2,w//2)
            
            # print(f'==========================batch:{b},step:{l}==========================')
            # print(f'pose:\n{pose}')
            observation = obs_all[l][b] #(h,w)
            _obsMask = mask_rotate(observation, pose[2], [int(pose[0]), int(pose[1])],h,w,h//2,w//2)
            
            '''print to test'''
            # print(f'Testing!!!!!!!!!!generate_maps_realPerfect()')
            # _linMask[int(pose[0]),int(pose[1])] = 9 
            # print(f'visible_mask:\n{_linMask}')
            # _obsMask[int(pose[0]),int(pose[1])] = 9
            # print(f'_obsMask:\n{_obsMask}')
            obsMask = np.logical_or(obsMask, _obsMask) #(h,w)
            # print(f'obsMask:\n{obsMask.astype(int)}')
            # linMask = np.logical_or(linMask, _linMask) #(h,w)
            # print(f'env_traj:{env_traj}')
            # map_semantic = env_traj * np.logical_and(linMask, obsMask).astype(np.int32)
            map_semantic = env_traj * obsMask.astype(np.int32)
            # print(f'map_semantic_before:{map_semantic}')
            # mask invisible layers
            map_invisible = (1 - obsMask).astype(np.int32)[np.newaxis,...]
            # map_semantic[0] = linMask * 1 - np.sum(map_semantic[1:],axis=0)
            # print(f'map_semantic_final:{np.sum(map_semantic,axis=0)}')

            if equal_invisible:
                unobserved_layer = np.ones((feature_dim, H, W),dtype=np.float32)/feature_dim
                map_semantic += unobserved_layer * map_invisible #(f,h,w)
            else:
                map_semantic = np.concatenate((map_invisible, map_semantic), axis=0) #(f+1,h,w)
            # mask out invisible areas
            # map_semantic = map_semantic * linMask
            # map_semantic = map_semantic
            maps_chnl.append(map_semantic) #Cls,H,W
            # maps_abst.append(map_updated)# H,W
        maps_channel.append(np.array(maps_chnl)) 
        # maps_abstract.append(np.array(maps_abst)) 
    maps_channel = np.transpose(np.array(maps_channel),(1,0,2,3,4)) #B,L,C,H,W -> L,B,C,H,W
    # maps_abstract = np.transpose(np.array(maps_abstract),(1,0,2,3)) #B,L,H,W -> L,B,H,W
    # print(f'maps:\n{observation_raw}')
    # print(f'masks:\n{masks}')
    return maps_channel, None

def range_round(x):
    '''get x into range [-pi,pi]
    '''
    if x>math.pi: 
        x-=2*math.pi
    elif x<-math.pi: 
        x+=2*math.pi
    return x

def minmax(x,y,x1,y1,x2,y2,bool_y=True,b=1):
    if bool_y:
        return (min(y1,y2)<y<max(y1,y2)) and (x*b>=0)
    else:
        return (min(x1,x2)<x<max(x1,x2)) and (y*b>=0)

def coord_in_range(pose,mask):
    '''judge if value is in view range = math.pi/2
        Regard the pose as origion, create a new coordinate system
        if index(x,y)-pose(x,y) is in FOV(the top left value of the grid), set mask value to 1, else 0

        input: pose (x,y,yaw)
        mask: H,W
    '''
    theta = math.pi/4
    assert theta*2 <= 0.5*math.pi

    #convert angle to gradient angle in xy_coordinate
    angle = pose[2] #counter-clockwise, positive x-axis go down and is 0 degree [0,2pi]
    k = angle - 0.5*math.pi #[-pi/2,3pi/2]
    # print('bef range',k,np.degrees(k))
    k = range_round(k) #[-pi,pi] positive x-axis is 0 degree
    # print('aft range',k,np.degrees(k))
    left_k = k + theta
    right_k = k - theta
    k1 = np.tan(left_k)
    k2 = np.tan(right_k)
    # dx dy based on tensor cordinate
    # 0.5 is half gird size, 1e-12 is to avid 0 division, -1e-4 to aoivd exceeding the boundary 
    d = min(abs(0.5/(np.sin(k)+1e-12))-1e-4,abs(0.5/(np.cos(k)+1e-12))-1e-4) 
    dx = np.sin(k) * d
    dy = -np.cos(k) * d
    x=np.array(pose[0]) + dx 
    y=np.array(pose[1]) + dy     
    
    # print(f'ori_x,ori_y:{pose[0]} {pose[1]} {np.degrees(angle)}')
    # print('dx,dy',dx,dy)
    # print('normal degree in 2d linear:',np.degrees(k))
    # print('(x,y):',x,y)
    #determine the way to filter coordinate in map
    judge=None

    if abs(k)<math.pi/4:
        judge = partial(minmax,bool_y=True,b=1)
        # print('bool_y=True,b=1')
    elif k > math.pi*3/4 or k < -math.pi*3/4:
        judge = partial(minmax,bool_y=True,b=-1)
        # print('bool_y=True,b=-1')
    elif math.pi/4 < k < math.pi*3/4:
        judge = partial(minmax,bool_y=False,b=1)
        # print('bool_y=False,b=1')
    elif -math.pi/4 > k > -math.pi*3/4:
        judge = partial(minmax,bool_y=False,b=-1)
        # print('bool_y=False,b=-1')
    else:
        pass

    # if isinstance(judge,None):
    #     print('judge type: ',type(judge))
    # print(f'k1:{k1}, left_k,k,right_k:{math.degrees(left_k),math.degrees(k),math.degrees(right_k)}')

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            #convert to relative index 
            #   x-downward, y-rightward, origin at top-left (tensor coordinate)
            #   x-rightward, y-upward, origin at pose
            x_index= (j-y) 
            y_index= -(i-x)

            if abs(k) == math.pi/4 :
                if np.tan(k) * y_index >= 0 and x_index >= 0:
                    mask[i,j] = 1
            elif abs(k) == math.pi*3/4:
                if np.tan(k) * y_index <= 0 and x_index <= 0:
                    mask[i,j] = 1
            else:
                x_bound_1 = np.array(y_index)/k1
                x_bound_2 = np.array(y_index)/k2
                if judge(x_index,y_index,x_bound_1,k1*x_index,x_bound_2,k2*x_index):
                    # print(f'ij:{i,j}    new_ij:{x_index,y_index}')
                    mask[i,j] = 1 #set value to 1 if this grid is visible
    # pose2 = pose
    # pose2[2] = math.degrees(pose[2])
    # print(f'pose:{pose2}')
    # print(f'mask:\n{mask}')
    return mask

def affineGrid(r, pose, mode="bilinear", pad_mode="zeros"):
    """
    rotate clockwise

    Inputs:
        r     - (c, h, w) Tensor
        pose  - (3) x,y,angle
    Outputs:
        r_rot - (h, w) Tensor
    """
    
    r = torch.tensor(r)[None,None,...] #(1, 1, h, w) 
    h, w = r.shape[2:]
    x = pose[0] / h * 2
    y = pose[1] / w * 2
    sin_t  = torch.sin(pose[2])
    cos_t  = torch.cos(pose[2])
    A      = torch.zeros(r.size(0), 2, 3)
    # rotate clockwise
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t
    A[:, 0, 2] = -x
    A[:, 1, 2] = -y

    # inverse pose to convert it back to center coordinate
    # A[:, 0, 0] = cos_t
    # A[:, 0, 1] = sin_t
    # A[:, 1, 0] = -sin_t
    # A[:, 1, 1] = cos_t
    # A[:, 0, 2] = x
    # A[:, 1, 2] = y
    # A = F.pad(A, (0,0,0,1), mode='constant') #(b, 3, 3)
    # A[:,2,2]=1
    # A = torch.inverse(A)[:,:2] #(b, 2, 3)
    
    grid   = F.affine_grid(A, r.size(), align_corners=False)
    r_rot  = F.grid_sample(r, grid,mode=mode, padding_mode=pad_mode, align_corners=False)    
    return r_rot[0,0].numpy()

def generate_dataset():
    args = get_args()
    p = Parameters(args)
    trainer=Observation_System(p,args)
    trainer.linear_generate()

def test_generate_map():
    x = 5
    y = 5
    print(f'original_(x,y):',x,y)
    for ya in [0,30,45,60,90,180,270,360]:
        print(f'yaw:{ya}')
        ya= np.radians(ya)
        pose = np.array([x,y,ya])[None,None,...]
        map = np.arange(0,121).reshape(11,11)
        res = generate_maps(pose,map)
    # print(res)

def split_batch(maxbatch,nsplit):
    '''
    Random pick '1' batch as test data every 'nsplit' batches
    from total 'maxbatch' batches

    input:
        maxbatch: int
        nsplit:   int
    output:
        train_index: 1d numpy array, (dim=maxbatch- maxbatch//nsplit)
        test_index:  1d numpy array  (dim=maxbatch//nsplit)
    '''
    #get total index
    _index = np.arange(maxbatch)
    nsize = maxbatch//nsplit
    index = _index.reshape(nsize,nsplit)
    #random select 1 every nsplit batches, and put into test_index
    randchoice = np.random.randint(0,nsplit,size=(nsize,1))
    test_index = gather_numpy(index, randchoice, dim=1).flatten()
    #get remaind batches as train_index
    train_index = np.setdiff1d(_index,test_index)
    print('\n','='*20,'Split dataset', '='*20)
    print(f'total_batch:{maxbatch}')
    print(f'test_index:{test_index.shape}\n{test_index}')
    print(f'train_index:{train_index.shape}\n{train_index}')

    return train_index, test_index

def gather_numpy(inputs, index, dim):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = inputs.shape[:dim] + inputs.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(inputs, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)

def test_affineGrid():
    obs = torch.zeros(1,5,6) #b,h,w
    obs[0][1][0] = 1
    pose = torch.tensor([2,3,torch.deg2rad(torch.tensor(45))])
    new_obs = affineGrid(obs,pose)
    print(f'-------test transform_affinegrid-----------')
    print(f'original obs:\n{obs}')
    print(f'new obs:\n{new_obs}')



if __name__ == '__main__':
    # train_once(0.1,'11x11-withwall_10_new',-1,1)
    # test_generate_map()
    generate_dataset()
    # test_affineGrid()
    # input_path = '2023_Feb15th_dataset_w_rgbd_raw'
    # data = np.load(f'{input_path}/slam_1/camera/1.npy',allow_pickle=True)
    # # print(data.shape)
    # image = rearrange(torch.tensor(data[5][1])[None,].float(),'b h w c -> b c h w')
    # print(image.shape)
    # # depth = data[5][2].reshape(480,640,3)
    # mode = 'bilinear' # interpolation mode for RGB images
    # rgb_downsampled = F.interpolate(image, scale_factor=0.25,
    #                                 mode=mode,
    #                                 align_corners=True)
    # rgb_downsampled = rearrange(asnumpy(rgb_downsampled), 'b c h w -> b h w c')
    # cv2.imwrite('5.jpg', rgb_downsampled[0])
    # cv2.imwrite('depth.jpg', depth*10) #x_rightward, y_upward, z_forward
    # print(image)
    # for i in range(480):
    #     for j in range(640):
    #         # if 100 < np.mean(image[i][j]) <200:
    #         if np.any(image[i][j]<80):
    #             print(i,j,image[i][j])
    # print(depth[324][167:170])
    # print(np.min(depth), np.max(depth))
    
    

    #system=Map_System()
    #system.gaussian_tf()
    #system=Observation_System()
    #system.image_generate(None)

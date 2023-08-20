import os,sys,copy,glob,time,math,torch,cv2,scipy.stats
import numpy as np
import matplotlib.pyplot as plt
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
'''
Gnereate data with different environment(map)
first_release:
    Sep19th 2022
Each file slice the former 100 waypoints

    March7th 2023
        Update to same as yolo code
Output:
    image_cls:      L,b,cls,obs_size,obs_size   
                    where cls = 1unknown+1ground+10obsjects
    delta:          L,b,3 (x,y,yaw) 
                    where yaw is in radian degree from 0 to 2pi
    maps:           L,b,cls,map_size,map_size
    map_labl:       map_size,map_size
    map_cls_labl    cls,map_size,map_size -convert map_labl to one hot tensor
'''
torch.set_printoptions(linewidth=4000,precision=3,threshold=1000000,sci_mode=False)
np.set_printoptions(linewidth=10000,threshold=100000,precision=3,suppress=True)
np.random.seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='Jul17th_perfect_12Layers')   
    # parser.add_argument('--date', type=str, default='Jun6th_perfect_RealObs_scale3')   
    parser.add_argument('--map_size', type=int, default=33,help='The size of environment')
    parser.add_argument('--map_scale', type=int, default=3,help='The scale of unit, 1meter can be devided by 3 units')
    parser.add_argument('--obs_size', type=int, default=33,help='The size of environment')
    parser.add_argument('--obs_threshold', type=float, default=0.1,help='threshold for valid observation')
    parser.add_argument('--obs_case', type=str, default='obstructed',help='perfect/obstructed/real')
    parser.add_argument('--n_object', type=int, default=41,help='Item quantities')
    parser.add_argument('--feature_dim', type=int, default=12, help='Item varieties + 1 ground per environment')
    parser.add_argument('--env_ctr', type=int, default=30, help='total environemtns')
    parser.add_argument('--tra_ctr', type=int, default=3, help='number of trajecotry per environment')
    parser.add_argument('--tra_len', type=int, default=80, help='number of steps per trajectory')
    parser.add_argument('--train_env', type=int, default=26, help='environment to train')
    parser.add_argument('--same_env', type=int, default=3, help='environment to train')
    parser.add_argument('--rot_mode', type=str, default='bilinear', help='bilinear or nearest')   
    args = parser.parse_args()
    args.out_folder = 'Jul17th_YoloSegment'
    args.out_path = f'/data1/mli170/2022_Sep20th_dataset/{args.out_folder}/'
    # args.input_path = f'/data/mli170/2022_Sep20th_dataset/map{args.map_size}_obj{args.n_object}'
    # args.input_path = '2023_Feb15th_dataset_w_rgbd_raw_scale3'
    # args.input_path = '2023_Apr29th_dataset_w_rgbd_raw_scale3'
    args.input_path = '2023_Jun19th_dataset_w_rgbd_raw_scale3_40objects'
    args.input_folder = 'data_3_seg_ground1'
    print(f'data_3_seg_ground1 has 1 in ground layer, 0 in invisible area, and each entry of layer is percentage of pixels for that label')
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
        # self.map_size = self.map_size + self.map_border
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
            # print(f'ori observation:\n{observation}')
            left,right,up,down = center[1], len(observation[0])-center[1]-1, center[0], len(observation)-center[0]-1
            observation = np.hstack((np.zeros((len(observation),y-left)),observation)) if left<y else observation if left==y else observation[:,(left-y):]
            observation = np.hstack((observation,np.zeros((len(observation),y-right)))) if right < y else observation  if right == y else observation[:,:(y-right)]
            observation = np.vstack((np.zeros((x-up,len(observation[0]))),observation)) if up < x else observation if up ==x else observation[up-x:,:]
            observation = observation = np.vstack((observation,np.zeros((x-down,len(observation[0]))))) if down < x else observation if down==x else observation[:(x-down),:]
            # print(f'cuted observation:\n{observation}')
            return observation

        def semantic_tf(observation):
            visible_dim = 11 
            observation_semantic = np.zeros((visible_dim,len(observation),len(observation[0])))
            for i in range(len(observation_semantic)):
                id = np.argwhere((observation == i))
                for j in id:
                    observation_semantic[i][j[0],j[1]]=1
                # sum = np.sum(observation_semantic[i])
                # if sum != 0:
                #     observation_semantic[i] /= sum
            # print(f'observation_semantic:{observation_semantic.shape}\n{observation_semantic}')
            return observation_semantic

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
                # print(f'i:{i}\t degree:{degree}')
                # print(f'bef_cut:{observation_temp}')

                # mask value 0 at invisible areas in each layer
                observation_temp = cut_observation(observation_temp,0)
                # print(f'aft_cut:{observation_temp}')
                observation[i] = observation_temp
            # print(f'rotate observation:\n{observation}')
            return observation

        def gen(env_start,env_end):
            x,y,z=[],[],[]
            pose_file_all=[]
            obs_file_all=[]
            obs_file_abst_all = []
            map_file_all=[]
            map_labl_file_all=[]
            map_semantic_file_all=[]

            # a=[]
            for j in tqdm(range(env_start,env_end+1)):
                pose_all,obs_all,obs_abst_all,env_all,env_seman_all=[],[],[],[],[]
                real_obs_all = []

                #get pose of one env
                env_map = copy.deepcopy(np.load(f'{self.args.input_path}/slam_{j}/label.npy',allow_pickle=True)) #h,w
                env_map =  F.interpolate(torch.tensor(env_map)[None,None,],scale_factor=self.args.map_scale,mode='nearest')[0,0].numpy()
                # print(f'Read map information:\n{env_map}')
                train_datas = []
                for k in range(1, 1 + self.args.tra_ctr):
                    # data = np.load(f'{self.args.input_path}/slam_{j}/data/{k}.npy',allow_pickle=True) #(L,3)
                    data = np.load(f'{self.args.input_path}/slam_{j}/{self.args.input_folder}/{k}.npy',allow_pickle=True) #(L,3)
                    # print(f"data:\n{data[k][0]}\n{data[k][1]}\n{data[k][2]}")
                    train_data = []
                    real_obs = []
                    for i in range(len(data)):
                        train_data.append(data[i][0]) #L,3
                        real_obs.append(np.array(data[i][2])) # L,f,H,W
                    train_datas.append(np.array(train_data)) # B,L,3
                    real_obs_all.append(np.array(real_obs)) # B, L f. H, W

                    env_all.append(env_map) #B,H,W
                    map_semantic = np.zeros((self.args.feature_dim,env_map.shape[0],env_map.shape[1]))
                    for dd in range(map_semantic.shape[0]):
                        id = np.argwhere(env_map == dd)
                        for jf in id:
                            map_semantic[dd][jf[0],jf[1]]=1
                    env_seman_all.append(map_semantic) #B,f,H,W
                    
                train_datas = np.array(train_datas) #(B,L,3)
                real_obs_all = np.array(real_obs_all) #(B,L,f,h,w)
                # print(f'train_datas:\n{train_datas.shape}\tsample: {train_datas[0][0]}')
                # print(f'real_obs_all:\n{real_obs_all.shape}\tsample: {real_obs_all[0][0]}')
                tra_ctr = 0
                #generate observation of given pose
                for b, r_data in enumerate(train_datas): #(B,L,3)
                    r_obs = []
                    abs_obs = []
                    ground_item_obs = []
                    valid_poses = []
                    # print(f'self.train_data:{train_data.shape}')
                    ctr = 0
                    tra_ctr += 1
                    # invalid_ctr = 0
                    # print(f'env_map:\n{env_map}')
                    for l, pose in enumerate(r_data): #(L,3)
                        # if ctr == 6: 
                        #     print(f"pose:{pose}")
                        #     print("-"*10,ctr,"-"*10)
                        #     print(f'env_all:\n{env_all[0]}')
                        ctr += 1
                        x, y = int(pose[0]*self.args.map_scale), int(pose[1]*self.args.map_scale)
                        index = [x,y]
                        angle_rd = pose[2] # rotate clockwise to x-positive axis
                        angle_dg = math.degrees(angle_rd)
                        # print('pose:',*pose[:2],angle_dg)
                        observation_raw = copy.deepcopy(env_map)
                        # print(f'map:\n{env_map}')
                        # observation_raw[index[0],index[1]] = -1
                        # print(f'pose:{index}{angle_dg}')
                        # print(f'ori observation:\n{observation_raw}')
                        observation = center_observation(observation_raw,index,self.p.half_local_map_size,self.p.half_local_map_size)
                        # print(f'step:{l}\t{index}{angle_dg}')
                        # print(f'center_observation observation:\n{observation}') 
                        observation_semantic = semantic_tf(observation)
                        # print(f'observation_semantic:{observation_semantic.shape}\n{observation_semantic}')
                        observation_semantic = rotate_3D(observation_semantic,angle_rd)
                        # print(f'perfect_observation_semantic:{observation_semantic.shape}\n{observation_semantic}')

                        # back_obs = rotate_3D(observation_semantic,-angle_rd)
                        # print(f'rotate back to test:{back_obs.shape}\n{back_obs}')

                        perfect_observation_semantic = observation_semantic #(f,h,w)
                        real_observation_semantic = real_obs_all[b,l] #(f,h,w)
                        # print(f'real_observation_semantic:{real_observation_semantic.shape}\n{real_observation_semantic}')
                        # print(f'and : {np.logical_and(perfect_observation_semantic, real_observation_semantic)}')
                    
                        if self.args.obs_case == 'perfect':
                            final_observation_semantic = perfect_observation_semantic
                        elif self.args.obs_case == 'real':
                            final_observation_semantic = real_observation_semantic
                        elif self.args.obs_case == 'obstructed':
                            final_observation_semantic = perfect_observation_semantic * real_observation_semantic.astype(bool)
                        else:
                            print(f'Observation case error! {self.args.obs_case} is not a valid observation case')
                        # print(f'final_visible:\n{final_observation_semantic}\n perfect:\n{perfect_observation_semantic}')
                        # print(f'perfect AND obstructed_perfect observation:\n{(final_observation_semantic==perfect_observation_semantic).astype(int)}')
                        ground_item_obs.append(np.sum(final_observation_semantic, axis=0))
                        unobserved_mask = (1 - (np.sum(final_observation_semantic, axis=0) > 0)).astype(np.int16)[np.newaxis,...] # (1,H,W)
                        final_observation_semantic = np.concatenate([unobserved_mask, final_observation_semantic])# f+1, H, W
                        # final_observation_semantic[0] = 1 - np.sum(final_observation_semantic[1:],axis=0)
                        # print(f'final_observation_semantic:{final_observation_semantic.shape}\n{final_observation_semantic}')

                        check = final_observation_semantic[2:] >= self.args.obs_threshold
                        if not (check.any() >= self.args.obs_threshold):
                            print(f"env:{j} trajecotry:{tra_ctr} step:{ctr} invalid observation found")
                            # print(f'perfect_observation_semantic:\n{perfect_observation_semantic}')
                            # print(f'real_observation_semantic:\n{real_observation_semantic}')
                            # print(f'obstructed_observation_semantic:\n{final_observation_semantic}')
                            # invalid_ctr += 1
                            # print(f'step:{ctr} has {invalid_ctr} invalid observations {check.shape}')
                            continue
                        # return  self.rotation_resample(observation_semantic) 
                        # if ctr == 7: print(f'observation_semantic:\n{observation_semantic.numpy()}')
                        # observation_semantic = F.normalize(observation_semantic,p=1,dim=0)
                        # if ctr == 7: print(f'observation_semantic normalize:\n{observation_semantic.numpy()}')
                        # return observation_semantic
                        r_obs.append(final_observation_semantic) #L,f,H,W
                        valid_poses.append(np.array(pose)) #L,f,H,W
                    # a.append(invalid_ctr)   
                    # print(f'valid_obs:{len(r_obs)}\tvalid_poses:{len(valid_poses)}')
                    r_obs = np.array(r_obs[:self.args.tra_len]) # slice valid steps
                    # print(f'r_obs:{r_obs.shape}')
                    obs_all.append(r_obs) # B,L,f,H,W
                    valid_poses = np.array(valid_poses[:self.args.tra_len]) # slice valid steps
                    pose_all.append(valid_poses)
                    
                # print(f'np.array(obs_all):{np.array(obs_all).shape}')
                obs_all = np.transpose(np.array(obs_all),(1,0,2,3,4)) # L,B,f,H,W
                pose_all = np.transpose(np.array(pose_all),(1,0,2)) # L,B,3
                maps_channel, maps_abstr = generate_maps(pose_all,env_map,self.args.feature_dim,self.args.map_scale) #L,B,f,H,W

                pose_file_all         = pose_all if len(pose_file_all)==0 else np.concatenate((pose_file_all,pose_all),axis=1) #delta (L,b,3)
                obs_file_all          = obs_all  if len(obs_file_all)==0 else np.concatenate((obs_file_all,obs_all),axis=1) #img_cls (L,b,cls,H,W)
                map_file_all          = maps_channel if len(map_file_all)==0 else np.concatenate((map_file_all,maps_channel),axis=1) #maps: (L,b,f,h,w)
                map_labl_file_all     = np.array(env_all) if len(map_labl_file_all)==0 else np.concatenate((map_labl_file_all,np.array(env_all)),axis=0) #env information: (b,h,w)
                map_semantic_file_all = np.array(env_seman_all) if len(map_semantic_file_all)==0 else np.concatenate((map_semantic_file_all,np.array(env_seman_all)),axis=0) #maps: (b,cls,h,w)
            # print(f'ctr_max_min:',max(a),min(a))
            for i in range(5):
                print(f'step:{i}')
                print(f'pose_file_all:\n{pose_file_all[i,0]}')
                print(f'obs_file_all:\n{obs_file_all[i,0,:5]}')
                print(f'map_file_all:\n{map_file_all[i,0,:5]}')
                print(f'map_labl_file_all:\n{map_labl_file_all[0]}')
                print(f'map_semantic_file_all:\n{map_semantic_file_all[0]}')
            return obs_file_all, pose_file_all, map_file_all, map_labl_file_all, map_semantic_file_all
       
       #========================Training and testing on different env==========================
        print('='*20,'Training and testing on different env', '='*20,'\n')
        print('-'*20,'training', '-'*20)
        obs, pos, maps, env_maps, env_smaps = gen(1,self.args.train_env)
        print(f'obs:{obs.shape}, \npos:{pos.shape}, \nmaps:{maps.shape}, \nenv_maps:{env_maps.shape}, \nenv_smaps:{env_smaps.shape}')
        np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_{self.args.train_env}of{self.args.env_ctr}env_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_train',
            image_cls=obs, delta=pos, maps=maps, map_labl=env_maps, map_cls_labl=env_smaps)

        print('-'*20,'testing ', '-'*20,'\n')
        print('-'*20,f'Env index {self.args.train_env+1}-{self.args.env_ctr}', '-'*20,'\n')
        obs, pos, maps, env_maps, env_smaps = gen(self.args.train_env+1,self.args.env_ctr)
        print('-'*20,'testing', '-'*20)
        remainder = self.args.env_ctr - self.args.train_env 
        print(f'obs:{obs.shape}, \npos:{pos.shape}, \nmaps:{maps.shape}, \nenv_maps:{env_maps.shape}, \nenv_smaps:{env_smaps.shape}\n')
        np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_{remainder}of{self.args.env_ctr}env_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_test',
            image_cls=obs, delta=pos, maps=maps, map_labl=env_maps, map_cls_labl=env_smaps)

        #========================Training and testing on same env==========================
        # print('\n','='*20,'Training and testing on same env but different trajecotry', '='*20)
        # obs, pos, maps, env_maps, env_smaps = gen(1,self.args.same_env)
        # print(f'obs:{obs.shape}, \npos:{pos.shape}, \nmaps:{maps.shape}, \nenv_maps:{env_maps.shape}, \nenv_smaps:{env_smaps.shape}')
        # B = pos.shape[1]
        # sample_length = 3 # for one env containg 30 tra, sample 30/sample_length=2 for testing
        # train_index, test_index = split_batch(B,sample_length) 
        # np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_{self.args.same_env}env_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_train',
        #     image_cls=obs[:,train_index,...], delta=pos[:,train_index,...], maps=maps[:,train_index,...], map_labl=env_maps[train_index,...], map_cls_labl=env_smaps[train_index,...])
        # np.savez(f'{self.args.out_path}/Gazebo_{self.args.date}_{self.args.same_env}env_map{self.args.map_size}_obj{self.args.n_object}_len{self.args.tra_len}_test',
        #             image_cls=obs[:,test_index,...], delta=pos[:,test_index,...], maps=maps[:,test_index,...], map_labl=env_maps[test_index,...], map_cls_labl=env_smaps[test_index,...])
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
            #t0 = timer()
            sin_t  = torch.sin(t)
            cos_t  = torch.cos(t)
            #t1 = timer()
            # This R convention means Y axis is downwards.
            A      = torch.zeros(r.size(0), 2, 3).to(device)
            #t2 = timer()
            A[:, 0, 0] = cos_t
            A[:, 0, 1] = sin_t
            A[:, 1, 0] = -sin_t
            A[:, 1, 1] = cos_t
            #t3 = timer()
            grid   = F.affine_grid(A, r.size(), align_corners=False)
            r_rot  = F.grid_sample(r, grid, mode=self.args.rot_mode, align_corners=False)
            #t4 = timer()
            # print("----------------------Rotate_tensor-----------------------",
            #     '{:<20s}: {:<10.3f}'.format('Torch_sin_cos',(t1-t0)),
            #     '{:<20s}: {:<10.3f}'.format('Torch_zero',(t2-t1)),
            #     '{:<20s}: {:<10.3f}'.format('Build_Rot_matrix',(t3-t2)),
            #     '{:<20s}: {:<10.3f}'.format('F_affine_grid_sample',(t4-t3)),
            #     sep='\n')

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

def generate_maps(pose_all,label_map,feature_dim,scale):
    '''
    Generate label map based on previous observation for each time step
    If the top left value of the grid is in FOV, the whole grid is regarded as in the FOV
    input: 
        pose_all L,B,3
        label_map H,W
    output:
        maps_semnatic: L,B,H,W
    '''
    observation_raw = copy.deepcopy(label_map)  #H,W
    pose_copy = copy.deepcopy(pose_all)
    maps_semnatic = []
    maps_abstract = []
    semantic=True
    #iterate over all poses
    for b in range(pose_copy.shape[1]):
        maps_b = []
        maps_a = []
        mask = np.zeros_like(observation_raw) #update mask for a trajecotry with L poses
        # print(f'start')
        for l in range(pose_copy.shape[0]): 
            pose = pose_copy[l][b]
            pose[0] *= scale
            pose[1] *= scale
            mask = coord_in_range(pose,mask)
            # print(f'pose:\n{pose_copy[l,b,0],pose_copy[l,b,1],math.degrees(pose_copy[l,b,2])}')
            maps = mask*observation_raw
            # print(f'pose:\n{pose_copy[l][b]}')
            # print(f'mask:\n{mask}')
            # print(f'maps:\n{maps}')
            maps_a.append(maps) # H,W

            map_semantic = np.zeros((feature_dim,label_map.shape[0],label_map.shape[1]))
            for i in range(map_semantic.shape[0]):
                id = np.argwhere(maps == i)
                for j in id:
                    map_semantic[i][j[0],j[1]]=1
            # print(f'map_semantic:\n{map_semantic}')
            # mask out invisible areas
            map_semantic = map_semantic * mask
            maps_b.append(map_semantic) #Cls,H,W

        maps_b=np.array(maps_b) #L,cls, H,W
        maps_a=np.array(maps_a)
        maps_semnatic.append(maps_b) 
        maps_abstract.append(maps_a)
    maps_semnatic = np.array(maps_semnatic)#B,L,C,H,W
    maps_semnatic = np.transpose(maps_semnatic,(1,0,2,3,4)) #L,B,C,H,W
    maps_abstract = np.array(maps_abstract)#B,L,C,H,W
    maps_abstract = np.transpose(maps_abstract,(1,0,2,3)) #L,B,H,W
    # print(f'maps:\n{observation_raw}')
    # print(f'masks:\n{masks}')
    return maps_semnatic, maps_abstract

def generate_maps_from_env_obs(pose_all, semantic_env, obs_all, feature_dim, scale, equal_invisible=False):
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
            _obsMask = mask_rotate(observation, np.rad2deg(pose[2]), [int(pose[0]), int(pose[1])],h,w,h//2,w//2)
            
            '''print to test'''
            # print(f'Testing!!!!!!!!!!generate_maps_realPerfect()')
            # _linMask[int(pose[0]),int(pose[1])] = 9 
            # print(f'visible_mask:\n{_linMask}')
            # _obsMask[int(pose[0]),int(pose[1])] = 9
            # print(f'_obsMask:\n{_obsMask}')
            # print(f'obsMask:\n{obsMask.shape}')
            obsMask = np.logical_or(obsMask, _obsMask) #(h,w)
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
    # print(np.degrees(k))
    # print(dx,dy)
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

def rotate_tensor(r, t, mode, pad_mode="zeros"):
    """
    rotate clockwise

    Inputs:
        r     - (h, w) Tensor
        t     - 1 Tensor of angles
    Outputs:
        r_rot - (h, w) Tensor
    """
    r = torch.tensor(r[None,None,...]) #(1, 1, h, w) 
    t = torch.tensor(t[None,...]) # (1, )
    sin_t  = torch.sin(t)
    cos_t  = torch.cos(t)
    # This R convention means Y axis is downwards.
    A      = torch.zeros(r.size(0), 2, 3)
    # rotate clockwise
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = sin_t
    A[:, 1, 0] = -sin_t
    A[:, 1, 1] = cos_t
    grid   = F.affine_grid(A, r.size(), align_corners=False)
    r_rot  = F.grid_sample(r, grid.double(),mode=mode, padding_mode=pad_mode, align_corners=False)
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

if __name__ == '__main__':
    # train_once(0.1,'11x11-withwall_10_new',-1,1)
    # test_generate_map()
    generate_dataset()

    #system=Map_System()
    #system.gaussian_tf()
    #system=Observation_System()
    #system.image_generate(None)

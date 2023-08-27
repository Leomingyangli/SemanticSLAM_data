import os,sys,copy,glob,time,math,torch,scipy.stats,json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import rotate
from einops import rearrange, reduce, asnumpy, parse_shape
# import open3d as o3d
# from mpl_toolkits.mplot3d import Axes3D
# from tqdm import trange



from parameters import Parameters
from segment_anthing_utils import show_mask,show_box

torch.set_printoptions(precision=2,linewidth=4000,threshold=1000000)
np.set_printoptions(precision=2,linewidth=1000000,threshold=10000000)


class Observation_System():
    def __init__(self,p, segment, device):
        self.p = p
        self.predictor = segment
        self.device = device
        #self.label_map = np.load(self.p.label_map_path,allow_pickle=True)
        #self.data = np.load(self.p.dataset,allow_pickle=True)
        self.grid_unit = self.p.grid_unit
        self.inf = 100*self.p.grid_count
        self.yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/best.pt',trust_repo=True)
        self.yolo.conf = 0.7
        self.ctr = 1

    def center_observation(self,observation,center,x,y):
        left,right,up,down = center[1], len(observation[0])-center[1]-1, center[0], len(observation)-center[0]-1
        observation = np.hstack((np.zeros((len(observation),y-left)),observation)) if left<y else observation if left==y else observation[:,(left-y):]
        observation = np.hstack((observation,np.zeros((len(observation),y-right)))) if right < y else observation  if right == y else observation[:,:(y-right)]
        observation = np.vstack((np.zeros((x-up,len(observation[0]))),observation)) if up < x else observation if up ==x else observation[up-x:,:]
        observation = np.vstack((observation,np.zeros((x-down,len(observation[0]))))) if down < x else observation if down==x else observation[:(x-down),:]
        return observation

    def semantic_tf(self,observation,normalize):
        observation_semantic = np.zeros((self.p.feat_dim_range[1],len(observation),len(observation[0])))
        for i in range(len(observation_semantic)):
            id = np.argwhere((observation == i))
            for j in id:
                observation_semantic[i][j[0],j[1]]=1
            if normalize:
                sum = np.sum(observation_semantic[i])
                if sum != 0:
                    observation_semantic[i] /= sum
        return observation_semantic[self.p.feat_dim_range[0]:self.p.feat_dim_range[1]]

    def cut_observation(self,observation):
        mask=np.zeros((len(observation),len(observation[0])))
        center = [int(len(observation)/2),int(len(observation[0])/2)]
        for i in range(center[0]+2):
            for j in range(i):
                mask[center[0]+1-i][center[1]-j] = 1
                mask[center[0]+1-i][center[1]+j] = 1
        for i in range(int(len(observation))):
            for j in range(int(len(observation[0]))):
                if(mask[i][j]==0):
                    observation[i][j]=0
        return observation

    def rotate_3D(self,observation,degree,cut):
        for i in range(len(observation)):
            observation_temp = rotate(observation[i], angle=degree,order=1,mode='constant')
            # index_new = [int(len(observation_temp)/2),int(len(observation_temp[0])/2)]
            # observation_temp = self.center_observation(observation_temp,index_new,self.p.half_local_map_size,self.p.half_local_map_size)
            # observation_temp = self.cut_observation(observation_temp) if cut else observation_temp
            observation[i] = observation_temp
        return observation

    def linear_generate(self,pose):
        index = [int(pose[0]),int(pose[1])]
        angle_rd = 3.1415-pose[2]
        angle_dg = math.degrees(angle_rd)
        observation_raw = copy.deepcopy(self.label_map)
        observation_raw[index[0],index[1]] = -1
        observation = self.center_observation(observation_raw,index,self.p.half_local_map_size,self.p.half_local_map_size)
        test = rotate(observation, angle=angle_dg+180,order=0,mode='constant')
        #print(test)
        index_new = [int(len(test)/2),int(len(test[0])/2)]
        test = self.center_observation(test,index_new,self.p.half_local_map_size,self.p.half_local_map_size)
        test[index_new[0],index_new[1]]=-1
        #test = cut_observation(test)
        #print(test)
        observation_semantic = self.semantic_tf(observation)
        observation_semantic = self.rotate_3D(observation_semantic,angle_dg+180,True)
        observation_semantic = torch.tensor(observation_semantic,dtype=torch.float32,device=self.p.device)[None,:]
        return  self.rotation_resample(observation_semantic)

    def image_generate(self,points,image=None):
        if type(image) != type(None):
            #self.p.view_rgb_image(image)
            image = image.reshape(-1,3)
            image = image/255
            image_t = np.transpose(image)
        points_t = np.transpose(points)
        for i in range(3):
            points_t[i] = np.where(points_t[i] == np.nan, self.inf, points_t[i])
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
                    observation[z[i],x[i],y_tf]=1
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
                    pass
                    #observation[i]/=sum
        else:
            pass
            #observation = np.divide(observation, observation_count, out=np.zeros_like(observation), where=observation_count!=0)

        observation = observation[self.p.feat_dim_range[0]:self.p.feat_dim_range[1]]
        '''
        observation_sum = torch.zeros((6, self.p.local_map_size, self.p.local_map_size), dtype=torch.float32,device=self.p.device)
        for i in range(2):
            for j in range(int(self.p.feat_dim_range[1]/6)):
                observation_sum[0+i] += observation[j*3+i*int(self.p.feat_dim_range[1]/2)]
                observation_sum[1+i] += observation[j*3+1+i*int(self.p.feat_dim_range[1]/2)]
                observation_sum[2+i] += observation[j*3+2+i*int(self.p.feat_dim_range[1]/2)]
        observation = torch.tensor(observation_sum,dtype=torch.float32,device=self.p.device)[None,:]
        '''
        observation = torch.tensor(observation,dtype=torch.float32,device=self.p.device)[None,:]
        '''
        size = observation.shape[1]*observation.shape[2]*observation.shape[3]
        count = 0
        for i in observation[0]:
            for j in i:
                for k in j:
                    if k!=0:
                        count+=1
        print(observation)
        print(count/size)
        input()
        '''
        return  self.rotation_resample(observation)

    def yolo_generate(self,image,pointcloud,position=0):
        def distance_sort(result):
            sorted_result=[]
            for pre in result:
                label = int(pre[5])
                aera = pointcloud[int(pre[1]):int(pre[3]),int(pre[0]):int(pre[2])]
                aera = aera.reshape((-1,3))
                aera_t = np.transpose(aera)
                x,y,z = aera_t[2],aera_t[0],aera_t[1]
                temp = pre.cpu().tolist()
                temp.append(np.sum(x)/len(x))
                sorted_result.append(temp)
            sorted_result = sorted(sorted_result, key=lambda x: x[-1])
            return sorted_result
        
        def filt_white(image,image_check):
            threshhold = 240
            for i in range(len(image)):
                for j in range(len(image[0])):
                    if image[i][j][0] >= threshhold and image[i][j][1] >= threshhold and image[i][j][2] >= threshhold:
                        image_check[i][j] = 1
            return image_check

        label_map = {'1':1,'2':7,'3':8,'4':9,'5':10,'6':2,'7':3,'8':4,'9':5,'10':6}

        # ------------get rgbd image and initialize ground observations-----------------
        # image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # pointcloud = pointcloud.reshape(image.shape)
        observation = np.zeros((self.p.local_map_size,self.p.local_map_size))
        # print(f'Initializing: observation:{observation.shape}')

        # ===================Yolo generate bounding box =================================
        results = self.yolo([image])
        # list of boundbox coordinates (xmin,ymin,xman,ymax,score,label) 
        # where y-axis downward
        result = results.xyxy[0] 
        if not len(result):
            return False, None, None
        # print(f'yolo_result:{result}')
        #results.show()
        #print(results.pandas().xyxy[0])
        #input()
        # result = distance_sort(result) # sort the box from nearest to farest

        # image_check = np.zeros((480,640))
        #image_check = filt_white(image,image_check)

        observation_semantic = np.zeros((11,self.p.local_map_size,self.p.local_map_size))
        observation_semantic_score = np.zeros((11,self.p.local_map_size,self.p.local_map_size))

        result = torch.tensor(result,device=self.device)
        input_boxes = result[:,:4] #B,4
        labels = result[:,5].cpu().numpy().astype(np.int32)#B
        self.predictor.set_image(image)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        # print(f'yolo_box:{input_boxes}')
        # print(f'segment_box:{transformed_boxes}')
        
        masks, _, _ = self.predictor.predict_torch(point_coords=None,
                                    point_labels=None,
                                    boxes=transformed_boxes,
                                    multimask_output=False,) #(L,1,H,W) 1-beacuse of False
        masks = masks.squeeze(1).cpu().numpy()

        '''Sample images output'''
        # plt.figure(figsize=(10, 10))
        # position[2] = np.rad2deg(position[2])
        # plt.title(f'step:{self.ctr} pose_x:{position[0]:.2f} y:{position[1]:.2f} yaw:{position[2]:.2f}',fontsize=20)
        # plt.imshow(image)
        # for i,mask in enumerate(masks):
        #     show_mask(mask, plt.gca(), random_color=True)
        # for input_box in input_boxes:
        #     show_box(input_box.cpu().numpy(), plt.gca())
        # plt.axis('off')
        # plt.savefig(f'sample_image_Aug21th/{self.ctr}.png')
        # self.ctr += 1
        # plt.clf()
        # plt.close()
        # return False, None, None

        # print(f'labels:{len(labels)}{labels}')
        ground_mask = np.zeros(image.shape[:2])
        for mask,label in zip(masks,labels):
            ground_mask = np.logical_or(ground_mask, mask)
            #pointcloud, H,W,3    mask H,W
            #value of masked pointCloud are 0
            # print(f'mask:{mask.shape}\tlabel:{label}')
            area = pointcloud[np.where(mask)] # b,3
            # print(f'total pointcloud:{area.shape}')
            aera_t = np.transpose(area) #3,b

            # x is forward distance(positive), y is rightward distance(real)
            x,y,z = aera_t[2],aera_t[0],aera_t[1] #b

            x = np.nan_to_num(x,nan=self.inf)
            y = np.nan_to_num(y,nan=self.inf)
            # print(f'x_min:{min(x)} x_max{max(x)}, y_min:{min(y)} y_max{max(y)}')
            # scale to observation resolution, then convert to index
            # e.g. when scale==1, distacne -0.5-0.5 is 0 cell change, 0.5-1.5 is 1 cell change.
            x = (x/self.grid_unit)
            y = (y/self.grid_unit)
            x = np.round(x).astype(np.int32)
            y = np.round(y).astype(np.int32)

            observation_semantic_score_temp = np.zeros((11,self.p.local_map_size,self.p.local_map_size))

            #Project pointCloud onto the ground
            # Convert distance to pixel in observation coordinates
            # print(f"self.p.half_local_map_size:{self.p.half_local_map_size}")
            ctr = 0
            for i in range(len(x)):
                if abs(y[i])<=self.p.half_local_map_size and x[i]<=self.p.half_local_map_size: # 
                    y_tf = self.p.half_local_map_size - y[i]
                    if observation[x[i],y_tf] == 0 or observation[x[i],y_tf] == label_map[str(label+1)]:
                        ctr += 1
                        observation[x[i],y_tf] = label_map[str(label+1)]
                        observation_semantic_score_temp[label_map[str(label+1)],x[i],y_tf] += 1
            # print(f'observation:\n{observation.astype(np.int32)}')
            # print(f'valid point cloud:{ctr}')
            # print(f'observation_semantic_score_temp:\n{observation_semantic_score_temp[label_map[str(label+1)]].astype(np.int32)}')
            #Generate semantic layers
            if ctr != 0:
                observation_semantic_score_temp[label_map[str(label+1)]] /= ctr
            observation_semantic_score += observation_semantic_score_temp
        else:
            ground_mask = ~ground_mask # other pixels are visible 'ground'
            area = pointcloud[np.where(ground_mask)] # b,3
            aera_t = np.transpose(area) #3,b
            x,y,z = aera_t[2],aera_t[0],aera_t[1] #b
            x = np.nan_to_num(x,nan=self.inf)
            y = np.nan_to_num(y,nan=self.inf)
            x = (x/self.grid_unit)
            y = (y/self.grid_unit)
            x = np.round(x).astype(np.int32)
            y = np.round(y).astype(np.int32)
            observation_semantic_score_temp = np.zeros((11,self.p.local_map_size,self.p.local_map_size))
            for i in range(len(x)):
                if abs(y[i])<=self.p.half_local_map_size and x[i]<=self.p.half_local_map_size:
                    y_tf = self.p.half_local_map_size - y[i]
                    if observation[x[i],y_tf] == 0:
                        observation[x[i],y_tf] = -1
                        observation_semantic_score_temp[0,x[i],y_tf] = 1
            observation_semantic_score += observation_semantic_score_temp
            
        # print(f'step:{i}observation:{observation.shape}\n{torch.tensor(observation)}')
        # for i in range(11):
        #     print(f'obserobservation_semantic_scorevation:{i}{observation_semantic_score.shape}\n{torch.tensor(observation_semantic_score[i])}')
        
        '''
        for pre in result:
            xmin,ymin,xmax,ymax = int(pre[0]),int(pre[1]),int(pre[2]),int(pre[3]) 
            label = int(pre[5])

            aera = pointcloud[ymin:ymax,xmin:xmax]
            aera_check = image_check[ymin:ymax,xmin:xmax]
            aera = aera[aera_check==0]
            image_check[ymin:ymax,xmin:xmax]=1

            if aera.shape[0]==0:
                continue

            aera_t = np.transpose(aera)
            x,y,z = aera_t[2],aera_t[0],aera_t[1]

            x = np.nan_to_num(x,nan=self.inf)
            y = np.nan_to_num(y,nan=self.inf)
            x = (x/self.grid_unit)
            y = (y/self.grid_unit)
            x = np.round(x).astype(np.int32)
            y = np.round(y).astype(np.int32)
            x_min = np.min(x)

            observation_semantic_score_temp = np.zeros((11,self.p.local_map_size,self.p.local_map_size))


            for i in range(len(x)):
                if (x[i]==x_min or x[i]==x_min+1) and abs(y[i])<=self.p.half_local_map_size and x[i]<=self.p.half_local_map_size:
                    y_tf = self.p.half_local_map_size - y[i]
                    if observation[x[i],y_tf] == 0 or observation[x[i],y_tf] == label_map[str(label+1)]:
                        observation[x[i],y_tf] = label_map[str(label+1)]
                        observation_semantic_score_temp[label_map[str(label+1)],x[i],y_tf] += 1

            for i in range(len(observation_semantic_score_temp)):
                if np.sum(observation_semantic_score_temp[i])!=0:
                    observation_semantic_score_temp[i] /= np.sum(observation_semantic_score_temp[i])
            observation_semantic_score += observation_semantic_score_temp
        '''

        #print(observation_semantic_score)
        # print('before cut')
        # print(observation.astype(np.int32))
        observation = self.center_observation(observation,[0,self.p.half_local_map_size],self.p.half_local_map_size,self.p.half_local_map_size)
        for i in range(len(observation_semantic_score)):
            observation_semantic_score[i] = self.center_observation(observation_semantic_score[i],[0,self.p.half_local_map_size],self.p.half_local_map_size,self.p.half_local_map_size)
        # print('after cut')
        # print(observation.astype(np.int32))
        #input()

        #observation = rotate(observation, angle=180,order=0,mode='constant')
        #observation_semantic_score = self.rotate_3D(observation_semantic_score,180,False)[self.p.feat_dim_range[0]:self.p.feat_dim_range[1]]
        # observation_semantic_score = observation_semantic_score[self.p.feat_dim_range[0]:self.p.feat_dim_range[1]]
        # observation_semantic = self.semantic_tf(observation,False)
        # observation_semantic *= observation_semantic_score * self.p.grid_count

        observation_semantic = torch.tensor(observation_semantic_score,dtype=torch.float32,device=self.p.device)
        valid = True if torch.sum(observation_semantic) > 0 else False
        # print(f'observation_after:\n{observation.astype(np.int32)}')
        # np.set_printoptions(precision=3,linewidth=4000,threshold=1000000)
        # print(f'observation_semantic_score:\n{observation_semantic_score.astype(np.float16)}')
        return valid, observation, observation_semantic

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
            r_rot  = F.grid_sample(r, grid, align_corners=False)
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

if __name__ == '__main__':
    obs_system = Observation_System(Parameters())
    data = np.load('../dataset/slam_1/raw/%d.npy'%(1),allow_pickle=True,encoding='bytes')
    sample = data[0]
    image = sample[1]
    point_cloud = sample[2]
    valid,observation,observation_s = obs_system.yolo_generate(image,point_cloud)
    torch.set_printoptions(precision=2,sci_mode=False)
    np.set_printoptions(precision=1)
    print(valid)
    print((observation))
    print((observation_s))
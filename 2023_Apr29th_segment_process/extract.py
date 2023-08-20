import os,json,torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import cv2
import shutil

from observation_system import Observation_System
from parameters import Parameters
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    np.set_printoptions(precision=2,linewidth=4000,threshold=1000000)
    torch.set_printoptions(precision=2,linewidth=4000,threshold=1000000)
    # print('''2023 Jun19th dataset
    #         version details: The ground layers are 1 in semantc and -1 in abstract, and invisible are all 0
    #         However, the pointcloud just points to farset points. So we use ceiling to indicate visible ground area.
    #         2.recommand set noise threshold to 0.02 when scale=3, 
    #         which means any cell contains 2% total pixel are considered as noise''')
    print('''2023 Jun19th dataset
            ground 1 and invisible are 0''')
    length = 80
    # output_path = 'data_1_seg'
    output_path = 'data_5_seg_ground1'
    print('intput_folder:','2023_Jun19th_dataset_w_rgbd_raw_scale3_40objects')
    print('output_folder:',output_path)
    # =============Load Segment Anything Model================
    device = 'cuda:0'
    # sam = sam_model_registry["vit_b"](checkpoint="../segment_anything/sam_vit_b_01ec64.pth")
    
    sam = sam_model_registry["vit_h"](checkpoint="../segment_anything/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    obs_system = Observation_System(Parameters(), predictor, device)
    
    
    for id in tqdm(range(1,31,1)):
        print('\nEnv id: slam_'+str(id))
        base_path='../2023_Jun19th_dataset_w_rgbd_raw_scale3_40objects/slam_%d/'%id
        create_folder(base_path+'/raw')
        #for i in range(3):
        #    shutil.move(base_path+"%d.npy"%(i+1), base_path+"/raw/%d.npy"%(i+1))

        files=os.listdir(base_path+'raw/')
        create_folder(base_path+'/image')
        #create_folder(base_path+'/data')
        #create_folder(base_path+'/data_2')
        create_folder(base_path+'/'+output_path)
        #create_folder(base_path+'/data_test')
        create_folder(base_path+'/camera')
        
        npy_files=[]
        for i in files:
            if i.find('.npy')>=0:
                npy_files.append(i)
        npy_files.sort()

        for i in npy_files:
            check=0
            name = i.split('.')[0]
            data = np.load(base_path+'raw/'+i,allow_pickle=True,encoding='bytes')

            # env = np.load(base_path+'label.npy',allow_pickle=True,encoding='bytes')
            # print(f'Env:\n{env}')
            n = 1
            tmp = []
            camera = []
            print(f'total data number:{len(data)}')
            for j in data:
                
                check+=1
                position = j[0]
                image = j[1].astype(np.int32)

                # print(image.shape)
                # plt.imsave(f'{check}.png',j[1])

                pointcloud = j[2].astype(np.float64)

                # ---------Call function to generate YoloV3+Segmentation observations---------------
                valid,observation,observation_s = obs_system.yolo_generate(image,pointcloud,np.array(position))
                if not valid: 
                    print('Invalid ID: ',check-1)
                    continue

                '''add ground layer to observations'''
                # ones = torch.ones((1,1,*observation_s.shape))
                # sum = torch.sum(observation_s,axis=1)
                # sum = ones - sum
                # sum = sum[0]
                # observation_s = observation_s[0]
                # observation_s = torch.vstack((sum,observation_s))
                # observation_s = observation_s[None,:]

                # print(f'observation_After:{observation.shape}\n{observation}')
                # print(f'obserobservation_semantic_scorevation_After:{observation_s.shape}\n{observation_s}')
                tmp.append(np.array([np.array(position),np.array(observation),np.array(observation_s)],dtype=object))
                #camera.append(np.array([np.array(position),np.array(image),np.array(pointcloud)],dtype=object))
                #image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                #cv2.imwrite(base_path+'/image/%s/%d.png'%(name,n),image)
                n+=1

                if n==length+1:
                    break

            
            if len(tmp) != length:
                print('Error length',f'{length} step needed but got {len(tmp)}')
                return
                num = length -len(tmp)
                for j in range(num):
                    tmp = np.vstack((tmp,tmp[-1]))

            # for j in range(length):
            #     tmp[j][2] = np.where(tmp[j][2]>0,tmp[j][2],0)
            #     tmp[j][2] = np.where(tmp[j][2]<1,tmp[j][2],1)
            print(f'observation_s:{observation_s.shape}')
            np.save(base_path+f'{output_path}/{name}.npy',np.array(tmp,dtype=object))
            #np.save(base_path+'camera/%s.npy'%(name),np.array(camera,dtype=object))
            #break

if __name__ == '__main__':
    main()

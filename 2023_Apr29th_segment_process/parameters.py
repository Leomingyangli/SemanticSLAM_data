import os,sys,cv2,torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Parameters():
    def __init__(self):
        self.device = torch.device('cpu')
        #Slam_System
        self.map_id = '1'
        self.previous_pose = [10,10]
        self.candidate = 1
        self.init_step = -1
        self.grid_unit = 1/5
        self.grid_count = int(1 /self.grid_unit)
        self.feature_unit = 1
        #

        #Map_System
        self.map_border = 2*2*self.grid_count
        self.map_size = 11*self.grid_count
        self.half_map_size = int(self.map_size/2)
        # self.map_size = self.map_size + self.map_border
        # self.map_size = self.map_size if self.map_size %2 ==1 else self.map_size+1
        self.local_map_size = 11 * self.grid_count
        self.local_map_size = self.local_map_size if self.local_map_size % 2 ==1 else self.local_map_size+1
        self.half_local_map_size = int(self.local_map_size/2)

        #self.feat_dim_range = (0,1*self.feature_unit*self.grid_count)
        self.feat_dim_range = (1,11)
        self.feat_dim = self.feat_dim_range[1] - self.feat_dim_range[0]
        self.map_scale = 1
        #self.map_shape = (6, self.map_size, self.map_size)
        #self.local_map_shape = (6, self.local_map_size, self.local_map_size)
        self.map_shape = (self.feat_dim, self.map_size, self.map_size)
        self.local_map_shape = (self.feat_dim, self.local_map_size, self.local_map_size)
        print(f'self.map_shape:{self.map_shape}')
        print(f'self.local_map_shape:{self.local_map_shape}')
        #

        self.nangles = 36
        self.angles = torch.Tensor(np.radians(np.linspace(0, 360, self.nangles + 1)[:-1]-180)).to(self.device)
        self.zero_angle_idx = self.nangles //2

        self.update_rate = 0.5
        self.threshold = 0.1

        self.dataset = '../data/5.1/world_1.npy'

        self.label_map_path = '../dataset/world_1/label.npy'
        self.train_data_path = '../data/5.23/env_1/train/1.npy'

    def get_pointcloud_from_depth(rgb,depth):
        rgb = o3d.geometry.Image(np.array(np.asarray(rgb)[:, :, :3]))
        depth = o3d.geometry.Image(np.array(np.asarray(depth)[:, :, :3]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,depth_scale =1, depth_trunc = 10, convert_rgb_to_intensity = False)
        cam = o3d.camera.PinholeCameraIntrinsic(640, 480, 402, 402, 320, 240)#W, H, Fx, Fy, Cx, Cy

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)
        points = np.asarray(pcd.points)*255/20
        points_t = np.transpose(points)
        #o3d.visualization.draw_geometries([pcd])
        points_im = points.reshape(480,640,3)

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

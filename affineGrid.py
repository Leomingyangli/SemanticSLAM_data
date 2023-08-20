from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import matplotlib
import os
import cv2
from PIL import Image 
from utils import *

torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
mode = 'nearest'
# img = pil_loader('/data1/mli170/2022_Sep20th_dataset/printlog/affineGridTest/image1.jpg')
img = (torch.arange(1,1*5*5+1)).view(1,5,5).float()
img = F.pad(img, (3,3,3,3), mode='constant')
# matplotlib.image.imsave("printlog/affineGridTest/obs.jpg", img.permute(1,2,0).numpy())
img = img[None,]  # img→tensor 3维→4维

print(f'img.shape: {img.shape}\n{img}')
for deg in range(0,45,5):
    print(f"================{deg} degrees=====================\n")
    poseStart = torch.tensor([5,5,0])
    deg = torch.tensor(deg)
    poseEnd = torch.tensor([8,6,deg.deg2rad()])
    poseChange = compute_relative_pose_ori(poseStart[None,], poseEnd[None,]).squeeze()
    print(f'poseChange: {poseChange} \tposeStart: {poseStart} \tposeEnd: {poseEnd}')

    h, w = img.shape[2:]
    t = poseChange[2]
    x = poseChange[0] / h * 2
    y = poseChange[1] / w * 2
    sin_t  = torch.sin(t)
    cos_t  = torch.cos(t)
    M = torch.tensor(                   # 仿射变换矩阵,先旋转，然后沿着新坐标系平移
        [[[cos_t, sin_t, x],
        [-sin_t, cos_t, y]]]
    )
    grid = F.affine_grid(M, img.size(),align_corners=False)
    warp_img = F.grid_sample(img, grid,mode=mode,align_corners=False)          # 扭转图片
    print(f'rotate:\n{warp_img}')

    # convert back
    poseReverse1 = compute_relative_pose_ori(poseEnd[None,], poseStart[None,]).squeeze()
    poseReverse2 = - poseChange
    print(f'M:{M.shape}\n{M}')
    M = F.pad(M, (0,0,0,1), mode='constant')
    M[:,2,2]=1
    poseReverse3 = torch.inverse(M)[:,:2]
    print(f'reverseM:{poseReverse3.shape}\n{poseReverse3}')
    ctr = 1
    for poseReverse in [poseReverse1, poseReverse2, poseReverse3]:
        if ctr <= 2:
            print(f'poseReverse: {poseReverse}')
            t = poseReverse[2]
            x = poseReverse[0] / h * 2
            y = poseReverse[1] / w * 2
            sin_t  = torch.sin(t)
            M = torch.tensor(                   # 仿射变换矩阵,先旋转，然后沿着新坐标系平移
                [[[cos_t, sin_t, x],
                [-sin_t, cos_t, y]]]
            )
            grid = F.affine_grid(M, warp_img.size(),align_corners=False)
            back_img = F.grid_sample(warp_img, grid,mode=mode,align_corners=False).squeeze(0)
        else:
            grid = F.affine_grid(poseReverse, warp_img.size(),align_corners=False)
            back_img = F.grid_sample(warp_img, grid,mode=mode,align_corners=False).squeeze(0)
        print(f'back_img_{ctr}:\n{back_img}')
        print(f'{torch.logical_or(img,back_img)}')

        ctr += 1
    # matplotlib.image.imsave(f"printlog/affineGridTest/obsRotate{deg}.jpg", warp_img.permute(1,2,0).numpy())






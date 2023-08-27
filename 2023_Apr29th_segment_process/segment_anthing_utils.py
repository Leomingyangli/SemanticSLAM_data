import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import torch
'''
Github link: https://github.com/facebookresearch/segment-anything

1.Install with pip:
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install opencv-python pycocotools matplotlib

2.Download model checkpoint to the current directory
    base model vit_b 357MB
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    large mdoel vit_h 2.4GB
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

3.image format
    ndarray in HWC uint8 format

4. time cost per image
    no prompt provided:
        vit_h costs around 5mins
        vit_b costs 2mins

    per prompt(e.g. one bounding box)
        vit_b 10s

5.below code is taken from
    https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
'''

def main():

    # ================Load model checkpoint and Input======================
    device = 'cuda:0'
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    # sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
    sam.to(device=device)
    image = cv2.imread('1.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = torch.tensor(image,device='cuda:0')
    print(f'image:{image.shape}')
    
    # ================If you generate mask for an entire image======================
    entire_image = False
    if entire_image:
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        print(masks)

    # ================Or if you have a prompt such as bounding box from yolo======================
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    # take a box as input, provided in xyxy format.
    input_boxes = torch.tensor([[300, 200, 350, 360],[200, 260, 250, 360],[200, 100, 250, 260]],device=device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(point_coords=None,
                                    point_labels=None,
                                    boxes=transformed_boxes,
                                    multimask_output=False,) # masks.shape  # (number_of_masks) x H x W
    print(f'mask:{masks.shape}\n{masks[0,0]}')   
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i,mask in enumerate(masks):
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # show_mask2(mask.cpu().numpy(), plt.gca(), colr=i)
    for input_box in input_boxes:
        show_box(input_box.cpu().numpy(), plt.gca())
    plt.axis('off')
    # plt.savefig('mask_L.png')
    plt.savefig('mask.png')


def index():
    a = torch.arange(4*8*3).view(4,8,3).numpy() #H,W
    b = torch.zeros(4,8)
    b[2,3] = 1
    b[0,6] = 1
    b = np.array(b,dtype=bool)
    # b = b[...,None]
    # res = a[b[0][:,None],b[1]]
    
    print(f'a:{a.shape}\n{a}')
    print(f'b:{b.shape}\n{b}')
    result = a[np.where(b)]
    print(f'result:{result.shape}\n{result}')
    # print(f'b:b.shape\n{b}')
    # print()
    # res = a * b
    # res = a[b]
    # print(f'res:\n{res}')



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask2(mask, ax, colr):

    color = np.array([(5*colr+1)/255, (10*colr+10)/255, 255/255, (colr+1)/10])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

if __name__ == "__main__":
    # main()
    index()
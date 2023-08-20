import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
# bs,c,seq = (1,2,3)
# inputs = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
# # Sample a batch of distributions. Usually this would come from the dataset
# target = F.softmax(torch.rand(3, 5), dim=1)
# loss = F.kl_div(inputs, target, reduction='batchmean')
# print(loss)
torch.set_printoptions(linewidth=4000,precision=3,threshold=1000000,sci_mode=False)
np.set_printoptions(linewidth=10000,threshold=100000,precision=2,suppress=True)

def test_dl_div():
    print('bs=1,c=2,seq=3 and cls layer is on channel c')
    a   = torch.tensor([[[0, 0, 0], [1, 1, 1]]]).float() # 1,2,3
    aa  = torch.tensor([[[0, 0, 0], [0.2, 1, 1]]]).float()
    aaa = torch.tensor([[[0, 0, 0], [0 , 1, 1]]]).float()
    # print(a.shape, aa.shape, aaa.shape,sep='\n')
    print('original input:', a, aa, aaa,sep='\n')
    a = F.softmax(a, dim=1)
    aa = F.softmax(aa, dim=1)
    aaa = F.softmax(aaa, dim=1)
    # print('softmax',a, aa, aaa, sep='\n')
    a = (a + 1e-12).log()
    aa = (aa + 1e-12).log()
    aaa = (aaa + 1e-12).log()
    # print('log', a, aa, aaa, sep='\n')
    a = rearrange(a, 'b c t -> (b t) c')
    aa = rearrange(aa, 'b c t -> (b t) c')
    aaa = rearrange(aaa, 'b c t -> (b t) c')
    # print(a.shape, aa.shape, aaa.shape)

    gt_a = rearrange(torch.tensor([[[0, 0, 0], [0, 1, 1]]]).float(), 'b c t -> (b t) c')
    gt_b = rearrange(torch.tensor([[[1, 0, 0], [0, 1, 1]]]).float(), 'b c t -> (b t) c')
    print(f'different combination of kl div')
    print(f'input a:\n{a},\ninput aa:\n{aa},\ninput aaa:\n{aaa}\noutput gt_a:\n{gt_a},\noutput gt_b:\n{gt_b}')
    print(f'if one entry at gt is 0 vector, how different input at corresponding position affect result?')
    loss1 = F.kl_div(a, gt_a, reduction='batchmean')
    loss2 = F.kl_div(aa, gt_a, reduction='batchmean')
    print('The answer is no influence However, the update rule should be same for all entries. ')

    print(f'a-gt_a: {loss1}\naa-gt_a: {loss2}')
    print(f'if entry at input is 0, how different output at corresponding position affect result?')
    loss3 = F.kl_div(aaa, gt_a, reduction='batchmean')
    loss4 = F.kl_div(aaa, gt_b, reduction='batchmean')
    print(f'aaa-gt_a: {loss3}\naaa-gt_b: {loss4}\n')


def test_softmax_under_yoloSeg():
    # a = torch.ones(3,12) * -1
    a = torch.zeros(5,12)
    #case1 part of observation on grid
    a[0,2] = 0.1

    #case2 multuple label w/o project1, noise 0.05, 0.01
    a[1,0], a[1,1] = 0.3,0.1
    a[1,4], a[1,5] = 0.05, 0.01

    #case3 multiple label with project 1
    a[2,0], a[2,1] = 1, 1
    a[2,4], a[2,5] = 1, 1

    #case4 100% confident ground or invisible
    a[3,0] = 1

    a[4,2] = 0.2

    aa = F.softmax(a,dim=1)
    aaa = F.softmax(a*10,dim=1)
    print(f'--case1 part of observation on grid\n{a[0]}')
    print(f'----after softmax:\n{aa[0]}')
    print(f'----after softmax with T=10:\n{aaa[0]}\n')

    print(f'--case2 multiple label w/o project1, assume noise 0.05, 0.01\n{a[1]}')
    print(f'----after softmax:\n{aa[1]}')
    print(f'----after softmax with T=10:\n{aaa[1]}\n')

    print(f'--case3 multuple label with project 1 from case 2 \n{a[2]}')
    print(f'----after softmax:\n{aa[2]}')
    print(f'----after softmax with T=10:\n{aaa[2]}\n')

    print(f'--case4 100% confident ground or invisible\n{a[3]}')
    print(f'----after softmax:\n{aa[3]}')
    print(f'----after softmax with T=10:\n{aaa[3]}\n')

    print(f'--case5 part of observation on grid\n{a[4]}')
    print(f'----after softmax:\n{aa[4]}')
    print(f'----after softmax with T=10:\n{aaa[4]}\n')
    
test_softmax_under_yoloSeg()
import numpy
import torch
from torch.nn import functional as F
import torch
from torch.nn import functional as F

def get_reliable_loss(cam, reliable_label, intermediate=True):
    device = torch.device('cuda:0')
    reliable_label = reliable_label.float()
    b, c, h, w = cam.size()
    with torch.no_grad():
        temp_mask = torch.ones(1,1,10,10).cuda().to(device)
        mask_temp = F.conv2d(reliable_label,temp_mask,padding=5)
    mask_temp = F.interpolate(mask_temp, size=(h, w))
    mask = (mask_temp ==100.)
    mask = mask.float()
    cam = F.softmax(cam,dim=1)
    loss = F.mse_loss(cam[:, :-1], mask)
    return loss





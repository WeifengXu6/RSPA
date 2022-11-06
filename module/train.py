import os

import torch
from torch.nn import functional as F

# from eps import get_eps_loss
from reliable_pam import get_reliable_loss
from util import pyutils





def max_onehot(x):
    n,c,h,w = x.size()#8 21 134 134
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def train_cls(train_loader, model, optimizer, max_step, args):

    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_loader)
    for iteration in range(args.max_iters):
        try:
            img_id, img, label = next(loader_iter)
        except:
            loader_iter = iter(train_loader)
            img_id, img, label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        pred = model(img)

        # Classification loss
        loss = F.multilabel_soft_margin_loss(pred, label)
        avg_meter.add({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (optimizer.global_step-1) % 50 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print('Iter:%5d/%5d' % (iteration, args.max_iters),
                  # 'Loss:%.4f' % (avg_meter.pop('loss')),
                  'Rem:%s' % (timer.get_est_remain()),
                  'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)


        timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))
def train_rspa(train_dataloader, model, optimizer, max_step, args):
    device = torch.device('cuda:0')
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls_all', 'loss_re_all','loss_er')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    for iteration in range(args.max_iters):
        try:
            img_id, img, reliable_label, label = next(loader_iter)
        except:
            loader_iter = iter(train_dataloader)
            img_id, img, reliable_label, label = next(loader_iter)
        img = img.cuda(non_blocking=True).to(device)
        reliable_label = reliable_label.cuda(non_blocking=True).to(device)
        label = label.cuda(non_blocking=True).to(device)
        pred, cam = model(img)


        loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)

        loss_re= \
            get_reliable_loss(cam, reliable_label,intermediate=True)


        scale_factor = 0.6
        img2 = F.interpolate(img, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        pred2, cam2 = model(img2)
        loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
        loss_re2 = \
            get_reliable_loss(cam2, reliable_label,intermediate=True)

        loss_cls_all = loss_cls2 + loss_cls
        loss_re_all = loss_re + loss_re2
        N2,C2,H2,W2 = cam2.size()
        cam = F.interpolate(cam, size=(H2,W2), mode='bilinear', align_corners=True)
        loss_er = torch.mean(torch.abs(cam[:, :-1, :, :] - cam2[:, :-1, :, :]))

        ############################################################

        loss = loss_cls_all + loss_re_all + loss_er

        avg_meter.add({'loss': loss.item(),
                       'loss_cls_all': loss_cls_all.item(),
                       'loss_re_all': loss_re_all.item(),
                       'loss_er': loss_er.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_ = str(iteration)
        if (optimizer.global_step-1) % 50 == 0:
            timer.update_progress(optimizer.global_step / max_step)


            print('Iter:%5d/%5d' % (iteration, args.max_iters),
                  'Rem:%s' % (timer.get_est_remain()),
                  'lr: %.6f' % (optimizer.param_groups[0]['lr']), flush=True)
        if (iteration ) % 1000 == 0:
            torch.save(model.module.state_dict(), args.log_folder + iter_ + '.pth')

        timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))


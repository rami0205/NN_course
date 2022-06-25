import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from torch._six import inf
import math

class NegativeCosSimLoss(nn.Module):
    def __init__(self):
        super(NegativeCosSimLoss, self).__init__()
        
    def forward(self, x, y):
        out = -torch.cosine_similarity(x, y, dim=1).mean()
        return out

class Denormalize(nn.Module):
    def __init__(self):
        super(Denormalize, self).__init__()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        
    def forward(self, img):
        return img*self.std+self.mean

def denormalize(img):
    std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
    mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
    return img*std+mean

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    
    return total

# cosine learning rate schdule
def adjust_learning_rate(optimizer, epoch, EPOCHS, cycle=None, warmup_epochs=5, init_lr=0.001, min_lr=1e-6):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if cycle is None: cycle = EPOCHS
    epoch %= cycle
    if epoch < warmup_epochs:
        lr = init_lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (init_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
            
def before_train(f, model, optim, batch_size, EPOCHS):
    total_params = num_parameters(model)
    f.write('==================ViT Encoder==================\n')
    f.write(f'height: {model.height}, width: {model.width}\n')
    f.write(f'patch height: {model.ph}, patch width: {model.pw}\n')
    f.write(f'dim_enc: {model.dim_enc}\n')
    f.write(f'dim_attn: {model.dim_attn}\n')
    f.write(f'num_heads: {model.num_heads}\n')
    f.write(f'dim_ffratio: {model.dim_ffratio}\n')
    f.write(f'num_blocks: {model.num_blocks}\n')
    f.write(f'total parameters: {format(total_params, ",")}\n')
    f.write('\n')
    f.write('======================etc======================\n')
    f.write(f'batch size: {batch_size}\n')
    f.write(f'epochs: {EPOCHS}\n')
    f.write(f'initial learning rate: {optim.param_groups[0]["lr"]}\n')
    f.write(f'optimizer weight decay: {optim.param_groups[0]["weight_decay"]}\n')
    f.write('\n')
    
def topk_right(tensor, target, k, dim):
    right = 0
    kpred = tensor.topk(k, dim=dim)[1]
    
    for i in range(len(tensor)):
        right += 1 if target[i].item() in kpred[i] else 0
    
    return right

def after_one_epoch(f, mode, epoch, EPOCHS, iteration, data_loader, epoch_loss, epoch_top1_acc, epoch_top5_acc, epoch_time, total_time, now):
    f.write(f'---------------------------------{mode}---------------------------------\n')
    f.write(f'epoch: {epoch+1}/{EPOCHS}, iter: {iteration+1}/{len(data_loader)}, {mode} loss: {epoch_loss:.4f}, \n')
    f.write(f'{mode} top-1 accuracy: {epoch_top1_acc*100:.4f}%, {mode} top-5 accuracy: {epoch_top5_acc*100:.4f}%, \n')
    f.write(f'epoch time: {epoch_time:.2f}, total time: {total_time:.2f}, epoch finish: {now}\n')
    if mode == 'valid': f.write('\n')
        
def sd_key_change(key_dict, value_dict):
    '''
    key_dict: (OrderedDict) state dict of the model loading pretrained model
    value_dict: (OrderedDict) state dict of the model pretrained to be loaded
    '''
    temp = OrderedDict()
    for key in key_dict.keys():
        _, value = value_dict.popitem(last=False)
        temp[key] = value
        
    return temp

def num_parameters(model):
    total_params = 0
    for i, param in enumerate(model.parameters()):
        params = 1
        for p in param.size():
            params *= p
        total_params += params
        
    return total_params


# ------------------------------------------------------------------------------------------------------------------------------------------------------------
def show_softmax_coding():
    x = torch.tensor([[0.1, 0.5, -0.4, 0.9],
                     [0.8, -0.2, 0.6, -0.7],
                     [0.9, 0.4, 0.2, -0.1]])
    y = torch.tensor([3, 1, 0])
    
    boonja = torch.exp(x)
    boonmo = torch.sum(torch.exp(x),dim=1).unsqueeze(1) # 각 batch 별로 exp 값을 합침

    print(f'softmax my coding:\n{boonja/boonmo}')
    print()
    print(f'softmax function:\n{F.softmax(x, dim=1)}')
    
def show_cross_entropy_loss_coding():
    x = torch.tensor([[0.1, 0.5, -0.4, 0.9],
                     [0.8, -0.2, 0.6, -0.7],
                     [0.9, 0.4, 0.2, -0.1]])
    y = torch.tensor([3, 1, 0])

    boonja = torch.exp(x)[torch.arange(len(x)), y] # 각 batch의 정답인 exp 값
    boonmo = torch.sum(torch.exp(x), dim=1) # 각 batch 별로 exp 값을 합침
    
    print(f'cross entropy loss my coding:\n{-torch.log(boonja/boonmo).mean()}')
    print()
    print(f'cross entropy loss function:\n{F.cross_entropy(x, y)}')
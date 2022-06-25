from tqdm.notebook import tqdm
import torch
from utils import adjust_learning_rate
from time import sleep
from utils import denormalize
import os
import matplotlib.pyplot as plt

def train_one_epoch(model, data_loader, accum_iter, lr_cycle, init_lr, min_lr, reconstruct_criterion, feature_criterion, 
                    loss_scaler, optimizer, device, epoch, total_epochs, TIME, max_norm=None):
    model.train()
    total_loss = 0
    total_reconstruct_loss = 0
    total_feature_loss = 0
    for data_iter, (origin, reducted) in enumerate(tqdm(data_loader)):
        if data_iter % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter / len(data_loader) + epoch, total_epochs, 
                                 cycle=lr_cycle, warmup_epochs=0, init_lr=init_lr, min_lr=min_lr)
        origin = origin.to(device)
        reducted = reducted.to(device)
        
        origin_out, enc_out, dec_out = model(origin, reducted)
        
        reconstruct_loss = reconstruct_criterion(dec_out, origin)
        feature_loss = feature_criterion(enc_out, origin_out)
        loss = reconstruct_loss + feature_loss*0.01
        
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter + 1) % accum_iter == 0)
        if (data_iter + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        total_loss += loss.item()*accum_iter
        total_reconstruct_loss += reconstruct_loss.item()
        total_feature_loss += feature_loss.item()
        
        if (data_iter+1)%100==0 or (data_iter+1)==len(data_loader):
            idx = torch.randint(0, origin.size(0), (1,)).item()
            train_plot(origin[idx], reducted[idx], dec_out[idx], TIME, epoch, data_iter)
            
        with open(f'./logs/{TIME}_train.txt', 'a') as f:
            f.writelines(f'epoch: [{epoch+1}/{total_epochs}], iter: [{data_iter+1}/{len(data_loader)}], loss: {total_loss/(data_iter+1):.4f} ')
            f.writelines(f'reconstruct loss: {total_reconstruct_loss/(data_iter+1):.4f}, feature loss: {total_feature_loss/(data_iter+1):.4f}\n')
        
    avg_loss = total_loss/len(data_loader)
    avg_reconstruct_loss = total_reconstruct_loss/len(data_loader)
    avg_feature_loss = total_feature_loss/len(data_loader)

def train_plot(origin, reducted, reconstructed, TIME, epoch, data_iter):
    os.makedirs(f'./train_plot/{TIME}', exist_ok=True)
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax1.imshow(denormalize(reducted.detach().cpu()).permute(1,2,0), aspect='auto')
    ax1.axis('off')
    ax2.imshow(denormalize(origin.detach().cpu()).permute(1,2,0), aspect='auto')
    ax2.axis('off')
    ax3.imshow(denormalize(reconstructed.detach().cpu()).permute(1,2,0), aspect='auto')
    ax3.axis('off')
    plt.savefig(f'./train_plot/{TIME}/{str(epoch+1).zfill(3)}-{str(data_iter+1).zfill(4)}.png', bbox_inches='tight', facecolor='white')
    plt.close()
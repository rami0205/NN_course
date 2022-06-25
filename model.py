import torch.nn as nn

class SRAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(SRAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, origin, reducted):
        origin_out = self.encoder(origin)
        enc_out = self.encoder(reducted)
        enc_out_ = enc_out.unsqueeze(-1).unsqueeze(-1)
        dec_out = self.decoder(enc_out_)
        
        return origin_out, enc_out, dec_out

class ReconstructConv(nn.Module):
    def __init__(self):
        super(ReconstructConv, self).__init__()
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 2),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 2, 2),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, 2),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(64, 3, 7, 7),
                                   nn.BatchNorm2d(3),
                                   nn.ReLU())
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        return x
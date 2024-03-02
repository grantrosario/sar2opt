import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import os
import time
import matplotlib.pyplot as plt
import pathlib
from IPython.display import clear_output
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
import cv2
from glob import glob
from torch_pconv import PConv2d

import matplotlib.pyplot as plt

from Discriminator import Discriminator

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.Ws = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.Wt = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.patch_size = 3
        self.conv1 = nn.Conv2d(36, 36, kernel_size=3, dilation=1, padding=1)  # 1152
        self.conv2 = nn.Conv2d(36, 36, kernel_size=3, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(36, 36, kernel_size=3, dilation=4, padding=4)
        self.conv8 = nn.Conv2d(36, 36, kernel_size=3, dilation=8, padding=8)
        self.weights = nn.Conv2d(36, 36*4, kernel_size=1)
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=2, padding=1)
        self.encoder_conv_depth = 16

        self.cfa_encoder_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.cfa_encoder_batchn1 = nn.BatchNorm2d(64)
        self.cfa_encoder_conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.cfa_encoder_batchn2 = nn.BatchNorm2d(32)
        self.cfa_encoder_conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.cfa_encoder_batchn3 = nn.BatchNorm2d(16)

        self.cfa_decoder_conv1 = nn.Conv2d(52, 32, kernel_size=3, stride=1, padding=1)
        self.cfa_decoder_batchn1 = nn.BatchNorm2d(32)
        self.cfa_decoder_conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.cfa_decoder_batchn2 = nn.BatchNorm2d(16)
        self.cfa_decoder_conv3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.cfa_decoder_batchn3 = nn.BatchNorm2d(3)

        # Texture Encoder
        self.t_encoder_pconv1 = PConv2d(3, 64, 7, 2, 3)
        self.t_encoder_pconv2 = PConv2d(64, 128, 5, 2, 2)
        self.t_encoder_pconv2_bn = nn.BatchNorm2d(128)
        self.t_encoder_pconv3 = PConv2d(128, 256, 5, 2, 2)
        self.t_encoder_pconv3_bn = nn.BatchNorm2d(256)
        self.t_encoder_pconv4 = PConv2d(256, 512, 3, 2, 1)
        self.t_encoder_pconv4_bn = nn.BatchNorm2d(512)
        self.t_encoder_pconv5 = PConv2d(512, 512, 3, 2, 1)
        self.t_encoder_pconv5_bn = nn.BatchNorm2d(512)
        self.t_encoder_pconv6 = PConv2d(512, 512, 3, 2, 1)
        self.t_encoder_pconv6_bn = nn.BatchNorm2d(512)
        self.t_encoder_pconv7 = PConv2d(512, 512, 3, 2, 1)

        # Structure Encoder
        self.s_encoder_pconv1 = PConv2d(2, 64, 7, 2, 3)
        self.s_encoder_pconv2 = PConv2d(64, 128, 5, 2, 2)
        self.s_encoder_pconv2_bn = nn.BatchNorm2d(128)
        self.s_encoder_pconv3 = PConv2d(128, 256, 5, 2, 2)
        self.s_encoder_pconv3_bn = nn.BatchNorm2d(256)
        self.s_encoder_pconv4 = PConv2d(256, 512, 3, 2, 1)
        self.s_encoder_pconv4_bn = nn.BatchNorm2d(512)
        self.s_encoder_pconv5 = PConv2d(512, 512, 3, 2, 1)
        self.s_encoder_pconv5_bn = nn.BatchNorm2d(512)
        self.s_encoder_pconv6 = PConv2d(512, 512, 3, 2, 1)
        self.s_encoder_pconv6_bn = nn.BatchNorm2d(512)
        self.s_encoder_pconv7 = PConv2d(512, 512, 3, 2, 1)

        # Texture Decoder
        self.t_decoder_pconv8 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.t_decoder_pconv8_bn = nn.BatchNorm2d(512)
        self.t_decoder_pconv9 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.t_decoder_pconv9_bn = nn.BatchNorm2d(512)
        self.t_decoder_pconv10 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.t_decoder_pconv10_bn = nn.BatchNorm2d(512)
        self.t_decoder_pconv11 = nn.ConvTranspose2d(768, 256, 3, 1, 1)
        self.t_decoder_pconv11_bn = nn.BatchNorm2d(256)
        self.t_decoder_pconv12 = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.t_decoder_pconv12_bn = nn.BatchNorm2d(128)
        self.t_decoder_pconv13 = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.t_decoder_pconv13_bn = nn.BatchNorm2d(64)
        self.t_decoder_pconv14 = nn.ConvTranspose2d(67, 64, 3, 1, 1)  # Texture Feature

        # Structure Decoder
        self.s_decoder_pconv15 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.s_decoder_pconv15_bn = nn.BatchNorm2d(512)
        self.s_decoder_pconv16 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.s_decoder_pconv16_bn = nn.BatchNorm2d(512)
        self.s_decoder_pconv17 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.s_decoder_pconv17_bn = nn.BatchNorm2d(512)
        self.s_decoder_pconv18 = nn.ConvTranspose2d(768, 256, 3, 1, 1)
        self.s_decoder_pconv18_bn = nn.BatchNorm2d(256)
        self.s_decoder_pconv19 = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.s_decoder_pconv19_bn = nn.BatchNorm2d(128)
        self.s_decoder_pconv20 = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.s_decoder_pconv20_bn = nn.BatchNorm2d(64)
        self.s_decoder_pconv21 = nn.ConvTranspose2d(66, 64, 3, 1, 1)  # Structure Feature

    def bi_gff(self, fs, ft):
        # BI-GFF Module
        # Apply Ws and Wt to concatenated features
        concatenated_features = torch.cat((fs, ft), dim=1)
        fs_prime = self.Ws(concatenated_features)
        ft_prime = self.Wt(concatenated_features)

        # Perform element-wise multiplication
        fs_prime = fs_prime * ft
        ft_prime = ft_prime * fs

        # Perform element-wise addition
        fs_out = fs + fs_prime
        ft_out = ft + ft_prime

        # Concatenate the modified features along the channel dimension to obtain the fused features
        f_out = torch.cat((fs_out, ft_out), dim=1)
        return f_out

    def cfa(self, F_in):
        F_in = self.cfa_encoder_batchn1(self.cfa_encoder_conv1(F.relu(F_in)))
        F_in = self.cfa_encoder_batchn2(self.cfa_encoder_conv2(F.relu(F_in)))
        F_in = self.cfa_encoder_batchn3(self.cfa_encoder_conv3(F.relu(F_in)))

        patches = self.unfold(F_in)  # shape: (batch_size, channels * patch_size * patch_size, L), L is the total number of patches
        patches = patches.permute(0, 2, 1).view(F_in.size(0), -1, self.encoder_conv_depth, self.patch_size, self.patch_size)

        # Compute cosine similarity between patches
        patches_normalized = F.normalize(patches, p=2, dim=2)  # normalize over the channel dimension
        similarity_matrix = torch.einsum('biklm,bjklm->bij', patches_normalized, patches_normalized)  # (batch_size, L, L)
        attention_scores = F.softmax(similarity_matrix, dim=-1)  # (batch_size, L, L)
        
        # Reconstruct the feature map from the attention scores and patches
        F_rec = torch.einsum('bij,bjklm->biklm', attention_scores, patches)  # (batch_size, L, C, patch_size, patch_size)
        F_rec = F_rec.view(F_in.size(0), -1, F_in.size(2), F_in.size(3))  # (batch_size, C, H, W)

        # Generate multiscale features with different dilation rates
        F1_rec = self.conv1(F_rec)
        F2_rec = self.conv2(F_rec)
        F4_rec = self.conv4(F_rec)
        F8_rec = self.conv8(F_rec)

        # Combine the multiscale features with the weights
        W = self.weights(F_rec)  # (batch_size, C*4, H, W)
        W1, W2, W4, W8 = torch.chunk(W, chunks=4, dim=1)  # split into four separate tensors

        # Final combination of features
        F_rec = (F1_rec * W1) + (F2_rec * W2) + (F4_rec * W4) + (F8_rec * W8)

        F_final = torch.cat((F_rec, F_in), dim=1) # skip connection
        F_final = self.cfa_decoder_batchn1(self.cfa_decoder_conv1(F.leaky_relu(F_final)))
        F_final = self.cfa_decoder_batchn2(self.cfa_decoder_conv2(F.leaky_relu(F_final)))
        F_final = self.cfa_decoder_batchn3(self.cfa_decoder_conv3(F.leaky_relu(F_final)))

        return F_final

    def feature_generator(self, t_encoder_inp, t_encoder_mask, s_encoder_inp_gray, s_encoder_inp_edge):
        # Texture Encoder
        t_e1, t_m1 = self.t_encoder_pconv1(t_encoder_inp, t_encoder_mask)
        t_e2, t_m2 = self.t_encoder_pconv2(F.relu(t_e1), t_m1)
        t_e2       = self.t_encoder_pconv2_bn(t_e2)
        t_e3, t_m3 = self.t_encoder_pconv3(F.relu(t_e2), t_m2)
        t_e3       = self.t_encoder_pconv3_bn(t_e3)
        t_e4, t_m4 = self.t_encoder_pconv4(F.relu(t_e3), t_m3)
        t_e4       = self.t_encoder_pconv4_bn(t_e4)
        t_e5, t_m5 = self.t_encoder_pconv5(F.relu(t_e4), t_m4)
        t_e5       = self.t_encoder_pconv5_bn(t_e5)
        t_e6, t_m6 = self.t_encoder_pconv6(F.relu(t_e5), t_m5)
        t_e6       = self.t_encoder_pconv6_bn(t_e6)
        t_e7, t_m7 = self.t_encoder_pconv7(F.relu(t_e6), t_m6)

        # Structure Encoder
        s_e1, s_m1 = self.s_encoder_pconv1(s_encoder_inp_gray, s_encoder_inp_edge)
        s_e2, s_m2 = self.s_encoder_pconv2(F.relu(s_e1), s_m1)
        s_e2     = self.s_encoder_pconv2_bn(s_e2)
        s_e3, s_m3 = self.s_encoder_pconv3(F.relu(s_e2), s_m2)
        s_e3     = self.s_encoder_pconv3_bn(s_e3)
        s_e4, s_m4 = self.s_encoder_pconv4(F.relu(s_e3), s_m3)
        s_e4     = self.s_encoder_pconv4_bn(s_e4)
        s_e5, s_m5 = self.s_encoder_pconv5(F.relu(s_e4), s_m4)
        s_e5     = self.s_encoder_pconv5_bn(s_e5)
        s_e6, s_m6 = self.s_encoder_pconv6(F.relu(s_e5), s_m5)
        s_e6     = self.s_encoder_pconv6_bn(s_e6)
        s_e7, s_m7 = self.s_encoder_pconv7(F.relu(s_e6), s_m6)

        # Texture Decoder
        s_e7_upsampled = F.interpolate(s_e7, size=(4, 4), mode='bilinear', align_corners=False)
        t_d1 = torch.cat([s_e7_upsampled, t_e6], 1)
        t_d1 = F.dropout(self.t_decoder_pconv8_bn(self.t_decoder_pconv8(F.leaky_relu(t_d1))), 0.5, training=True)
        
        t_d1_upsampled = F.interpolate(t_d1, size=(8, 8), mode='bilinear', align_corners=False)
        t_d2 = torch.cat([t_d1_upsampled, t_e5], 1)
        t_d2 = F.dropout(self.t_decoder_pconv9_bn(self.t_decoder_pconv9(F.leaky_relu(t_d2))), 0.5, training=True)
        
        t_d2_upsampled = F.interpolate(t_d2, size=(16, 16), mode='bilinear', align_corners=False)
        t_d3 = torch.cat([t_d2_upsampled, t_e4], 1)
        t_d3 = self.t_decoder_pconv10_bn(self.t_decoder_pconv10(F.leaky_relu(t_d3)))
        
        t_d3_upsampled = F.interpolate(t_d3, size=(32, 32), mode='bilinear', align_corners=False) 
        t_d4 = torch.cat([t_d3_upsampled, t_e3], 1)
        t_d4 = self.t_decoder_pconv11_bn(self.t_decoder_pconv11(F.leaky_relu(t_d4)))

        t_d4_upsampled = F.interpolate(t_d4, size=(64, 64), mode='bilinear', align_corners=False) 
        t_d5 = torch.cat([t_d4_upsampled, t_e2], 1)
        t_d5 = self.t_decoder_pconv12_bn(self.t_decoder_pconv12(F.leaky_relu(t_d5)))
        
        t_d5_upsampled = F.interpolate(t_d5, size=(128, 128), mode='bilinear', align_corners=False) 
        t_d6 = torch.cat([t_d5_upsampled, t_e1], 1)
        t_d6 = self.t_decoder_pconv13_bn(self.t_decoder_pconv13(F.leaky_relu(t_d6)))

        t_d6_upsampled = F.interpolate(t_d6, size=(256, 256), mode='bilinear', align_corners=False) 
        t_d7 = torch.cat([t_d6_upsampled, t_encoder_inp], 1)

        texture_feature = self.t_decoder_pconv14(F.leaky_relu(t_d7))

        # Structure Decoder
        t_e7_upsampled = F.interpolate(t_e7, size=(4, 4), mode='bilinear', align_corners=False)
        s_d1 = torch.cat([t_e7_upsampled, s_e6], 1)
        s_d1 = F.dropout(self.s_decoder_pconv15_bn(self.s_decoder_pconv15(F.leaky_relu(s_d1))), 0.5, training=True)

        s_d1_upsampled = F.interpolate(s_d1, size=(8, 8), mode='bilinear', align_corners=False)
        s_d2 = torch.cat([s_d1_upsampled, s_e5], 1)
        s_d2 = F.dropout(self.s_decoder_pconv16_bn(self.s_decoder_pconv16(F.leaky_relu(s_d2))), 0.5, training=True)

        s_d2_upsampled = F.interpolate(s_d2, size=(16, 16), mode='bilinear', align_corners=False)
        s_d3 = torch.cat([s_d2_upsampled, s_e4], 1)
        s_d3 = self.s_decoder_pconv17_bn(self.s_decoder_pconv17(F.leaky_relu(s_d3)))

        s_d3_upsampled = F.interpolate(s_d3, size=(32, 32), mode='bilinear', align_corners=False)
        s_d4 = torch.cat([s_d3_upsampled, s_e3], 1)
        s_d4 = self.s_decoder_pconv18_bn(self.s_decoder_pconv18(F.leaky_relu(s_d4)))

        s_d4_upsampled = F.interpolate(s_d4, size=(64, 64), mode='bilinear', align_corners=False)
        s_d5 = torch.cat([s_d4_upsampled, s_e2], 1)
        s_d5 = self.s_decoder_pconv19_bn(self.s_decoder_pconv19(F.leaky_relu(s_d5)))

        s_d5_upsampled = F.interpolate(s_d5, size=(128, 128), mode='bilinear', align_corners=False)
        s_d6 = torch.cat([s_d5_upsampled, s_e1], 1)
        s_d6 = self.s_decoder_pconv20_bn(self.s_decoder_pconv20(F.leaky_relu(s_d6)))

        s_d6_upsampled = F.interpolate(s_d6, size=(256, 256), mode='bilinear', align_corners=False)
        s_d7 = torch.cat([s_d6_upsampled, s_encoder_inp_gray], 1)

        structure_feature = self.s_decoder_pconv21(F.leaky_relu(s_d7))

        return texture_feature, structure_feature

    def forward(self, t_encoder_inp, t_encoder_mask, s_encoder_inp_gray, s_encoder_inp_edge):

        texture_feature, structure_feature = self.feature_generator(t_encoder_inp, 
                                                                    t_encoder_mask, 
                                                                    s_encoder_inp_gray, 
                                                                    s_encoder_inp_edge)

        F_in = self.bi_gff(structure_feature, texture_feature)
        F_out = self.cfa(F_in)

        return F_out, structure_feature, texture_feature


# if __name__== "__main__":

#     def cv2torchRGB(img):
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype("float32") / 255.0
#         img = img.transpose((2, 0, 1))
#         img = torch.tensor(img)
#         return img.unsqueeze(0)

#     def cv2torchGRAY(img):
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = img.astype("float32") / 255.0
#         #img = img.transpose((2, 0, 1))
#         img = torch.tensor(img)
#         if img.dim() == 2:
#             img = img.unsqueeze(0).unsqueeze(0)
#         else:
#             img = img.unsqueeze(1)
#         return img

#     sar_rgb = cv2.imread("/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_rgb_tiles/16.png")
#     sar_gray = cv2.imread("/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_gray_tiles/16.png")
#     sar_edge = cv2.imread("/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_edge_tiles/16.png")
#     opt_rgb = cv2.imread("/ssd/Datasets/sar2opt_raw_data/split_data/train/s2_rgb_tiles/16.png")

#     s_in_gray = cv2torchGRAY(sar_gray)
#     s_in_edge = cv2torchGRAY(sar_edge)
#     s_in = torch.cat((s_in_gray, s_in_edge), dim=1)
#     t_in_rgb = cv2torchRGB(sar_rgb)

#     batch_size, channels, height, width = t_in_rgb.size()
#     t_in_mask = torch.ones(batch_size, 1, height, width, dtype=t_in_rgb.dtype, device=t_in_rgb.device)

#     generator = Generator()
#     print(t_in_rgb.shape)
#     print(t_in_mask.squeeze(1).shape)


    #f = generator(t_in_rgb, t_in_mask.squeeze(1), s_in, s_in_edge.squeeze(1))
    # gen_out = f.detach().numpy()[0].T
    # discriminator = Discriminator()
    # r_probs, fake_probs = discriminator(opt_rgb, gen_out.astype('uint8'))
    # print(r_probs)
    # print("\n")
    # print(fake_probs)

    #plt.imshow(gen_out.astype('uint8'))
    #plt.show()



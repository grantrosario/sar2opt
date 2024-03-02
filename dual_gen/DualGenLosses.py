import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import Normalize

def focal_frequency_loss(pseudo_optical_image, real_optical_image, weight_matrix):
    # Assume both pseudo_optical_image and real_optical_image are of shape [batch_size, 1, H, W]
    # and weight_matrix is of shape [H, W]
    
    # Compute the 2D DFT for both images
    dft_pseudo = torch.fft.fft2(pseudo_optical_image)
    dft_real = torch.fft.fft2(real_optical_image)
    
    # Normalize the DFT values by dividing by sqrt(HW)
    H, W = pseudo_optical_image.size(-2), pseudo_optical_image.size(-1)
    norm_factor = torch.sqrt(torch.tensor(H * W))
    dft_pseudo_normalized = dft_pseudo / norm_factor
    dft_real_normalized = dft_real / norm_factor
    
    # Apply the dynamic spectral weight matrix
    # Expand weight_matrix to match the batch size and channels of the images
    weight_matrix = weight_matrix.to(pseudo_optical_image.device)
    weight_matrix_expanded = weight_matrix.unsqueeze(0).unsqueeze(0)
    weight_matrix_expanded = weight_matrix_expanded.expand_as(dft_pseudo_normalized)
    
    # Calculate the weighted DFT values
    weighted_dft_pseudo = dft_pseudo_normalized * weight_matrix_expanded
    weighted_dft_real = dft_real_normalized * weight_matrix_expanded
    
    # Compute the FFL loss
    ffl_loss = torch.mean((weighted_dft_pseudo - weighted_dft_real).abs().pow(2))
    
    return ffl_loss

def gram_matrix(features):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N images.
    """
    features_c = features.clone()
    N, C, H, W = features_c.size()
    features_c = features_c.view(N, C, H * W)
    gram = torch.matmul(features_c, features_c.transpose(1, 2))
    return gram / (C * H * W)


class VGGLoss(nn.Module):
    def __init__(self, feature_layers=[4, 9, 18], use_normalization=True, device='cuda'):
        super(VGGLoss, self).__init__()
        self.feature_layers = feature_layers

        # Load VGG19 with pre-trained ImageNet weights
        weights = VGG19_Weights.IMAGENET1K_V1
        self.vgg = vgg19(weights=weights).features[:max(feature_layers)+1].eval().to(device)

        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters

        # VGG networks are trained on images with each channel normalized
        # by mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        if use_normalization:
            self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
        else:
            self.normalize = None

    # Function to compute the Gram matrix for Style loss
    def gram_matrix(self, features):
        (b, c, h, w) = features.size()
        features = features.view(b, c, h * w)
        G = torch.mm(features, features.transpose(1, 2))
        return G.div(c * h * w)

    # Function to compute VGG loss
    def vgg_loss(self, pseudo_image, real_image):
        loss = 0.0
        pseudo_image = self.normalize(pseudo_image)
        real_image = self.normalize(real_image)
        for i in self.feature_layers:
            features_pseudo = self.vgg[:i](pseudo_image)
            features_real = self.vgg[:i](real_image)
            loss += F.l1_loss(features_pseudo, features_real)
        return loss

    # Function to compute Style loss
    def style_loss(self, pseudo_image, real_image):
        loss = 0.0
        pseudo_image = self.normalize(pseudo_image)
        real_image = self.normalize(real_image)
        for i in self.feature_layers:
            features_pseudo = self.vgg[:i](pseudo_image)
            features_real = self.vgg[:i](real_image)
            G_pseudo = gram_matrix(features_pseudo)
            G_real = gram_matrix(features_real)
            loss += F.l1_loss(G_pseudo, G_real)
        return loss

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from torchvision.utils import save_image
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Discriminator import Discriminator
from Generator import Generator
from Sar2optData import SAR2OpticalDataset
from DualGenLosses import focal_frequency_loss, VGGLoss
from utils import StabilizationChecker

import matplotlib.pyplot as plt


# Hyperparameters
num_epochs = 1000
data_subset_size = 2100
batch_size = 1
initial_lr_discriminator = 0.0002
initial_lr_generator = initial_lr_discriminator * 10
stabilization_threshold = 25
height = 256
width = 256
sar_rgb_dir = "/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_rgb_tiles"
sar_edge_dir = "/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_edge_tiles"
sar_gray_dir = "/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_gray_tiles"
optical_rgb_dir = "/ssd/Datasets/sar2opt_raw_data/split_data/train/s2_rgb_tiles"
optical_edge_dir = "/ssd/Datasets/sar2opt_raw_data/split_data/train/s2_edge_tiles"

# Directory where you want to save your models
save_dir = "/ssd/Datasets/sar2opt_raw_data/saved_models"


device = torch.device("cuda:0")
print(f"Using device: {device}")

# Loss functions
adversarial_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
rec_weight_matrix = torch.nn.Parameter(torch.ones((height, width), dtype=torch.float32, device=device), requires_grad=True)
texture_weight_matrix = torch.nn.Parameter(torch.ones((height, width), dtype=torch.float32, device=device), requires_grad=True)

generator = Generator().to(device)
generator_parameters = list(generator.parameters()) + [rec_weight_matrix]  + [texture_weight_matrix]

discriminator = Discriminator().to(device)

# optimizers
g_optimizer = optim.Adam(generator_parameters, lr=initial_lr_generator)
d_optimizer = optim.Adam(discriminator.parameters(), lr=initial_lr_discriminator)


# DataLoader
dataset = SAR2OpticalDataset(sar_rgb_dir  = sar_rgb_dir,
							 sar_edge_dir = sar_edge_dir,
							 sar_gray_dir = sar_gray_dir,
							 optical_rgb_dir = optical_rgb_dir,
							 optical_edge_dir = optical_edge_dir)

# Generate random indices for the subset
subset_indices = torch.randperm(len(dataset))[:data_subset_size]

# Create a SubsetRandomSampler to sample from the subset
subset_sampler = SubsetRandomSampler(subset_indices)

dataloader = DataLoader(dataset, batch_size=batch_size, sampler=subset_sampler)

# VGG Initiation for vgg/style loss
vgg_loss = VGGLoss(device=device)

# Create an instance of the checker with the desired number of epochs to track
stabilization_checker = StabilizationChecker(num_epochs_to_track=15)

def freeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

def get_validation_input(sar_rgb, sar_edge, sar_gray, optical):
	sar_rgb_img = Image.open(sar_rgb).convert('RGB')
	sar_edge_img = Image.open(sar_edge).convert('L') # grayscale
	sar_gray_img = Image.open(sar_gray).convert('L') # grayscale
	optical_rgb_img = Image.open(optical).convert('RGB')

	transformed_imgs = []
	for img in [sar_rgb_img, sar_edge_img, sar_gray_img, optical_rgb_img]:
		img = Resize((256, 256))(img)
		img = ToTensor()(img)
		if img.shape[0] == 3:
			img = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
		elif img.shape[0] == 1:
			img = Normalize(mean=[0.5], std=[0.5])(img)
		transformed_imgs.append(img)

	sar_rgb_img     = transformed_imgs[0]
	sar_edge_img    = transformed_imgs[1]
	sar_gray_img    = transformed_imgs[2]
	optical_rgb_img = transformed_imgs[3]

	return sar_rgb_img, sar_edge_img, sar_gray_img, optical_rgb_img

generator_checkpoints = sorted([ckpt for ckpt in os.listdir(save_dir) if "generator" in ckpt])
discriminator_checkpoints = sorted([ckpt for ckpt in os.listdir(save_dir) if "discriminator" in ckpt])

if generator_checkpoints:
	print("Loading previous checkpoint!\n")

	# Load the last checkpoint (or select the best checkpoint based on your criteria)
	last_generator_checkpoint = generator_checkpoints[-1]
	last_discriminator_checkpoint = discriminator_checkpoints[-1]

	generator.load_state_dict(torch.load(os.path.join(save_dir, last_generator_checkpoint), map_location=device))
	generator.train()

	discriminator.load_state_dict(torch.load(os.path.join(save_dir, last_discriminator_checkpoint), map_location=device))
	discriminator.train()



# Training loop
for epoch in range(0, num_epochs, 1):
	epoch_start_time = time.time()
	for i, (sar_rgb, sar_edge, sar_gray, true_optical, true_edge, true_gray) in tqdm(enumerate(dataloader)):
		batch_start_time = time.time()

		sar_rgb      = sar_rgb.to(device)
		sar_edge     = sar_edge.to(device)
		sar_gray     = sar_gray.to(device)

		sar_structure_in   = torch.cat((sar_gray, sar_edge), dim=1)
		sar_structure_mask = sar_edge.squeeze(1)
		sar_texture_mask   = torch.ones(batch_size, 1, height, width, dtype=sar_rgb.dtype, device=sar_rgb.device).squeeze(1)

		pseudo_optical, pseudo_structure_feature, pseudo_texture_feature = generator(sar_rgb, 
																					 sar_texture_mask, 
																					 sar_structure_in, 
																					 sar_structure_mask)
		pseudo_optical_detached = pseudo_optical.detach().clone()


		true_optical = true_optical.to(device)
		true_edge    = true_edge.to(device)
		true_gray    = true_gray.to(device)

		# optical_structure_in   = torch.cat((true_gray, true_edge), dim=1)
		# optical_structure_mask = true_edge.squeeze(1)
		# optical_texture_mask   = torch.ones(batch_size, 1, height, width, dtype=sar_rgb.dtype, device=sar_rgb.device).squeeze(1)

		# real_texture_feature, real_structure_feature = generator.feature_generator(true_optical, 
		# 													 					   optical_texture_mask, 
		# 																		   optical_structure_in, 
		# 																		   optical_structure_mask)

		real_structure_feature = true_edge
		real_texture_feature = true_optical

		pseudo_structure_feature = pseudo_structure_feature.view(1, 1, -1, 256, 256).mean(dim=2)
		weights = torch.randn(3, 64).to(device)
		weights = weights.view(3, 64, 1, 1)
		tensor_expanded = pseudo_texture_feature.repeat(1, 3, 1, 1, 1)
		pseudo_texture_feature = (tensor_expanded * weights).sum(dim=2)


		real_edges   = []
		pseudo_edges = []
		for i in range(batch_size):
			real_image_np = true_optical[i].cpu().numpy().transpose(1, 2, 0)
			real_image_np = (real_image_np * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
			real_edge = cv2.GaussianBlur(real_image_np, (3,3), 0)
			real_edge = cv2.Canny(real_edge, 30, 150)
			real_edge = np.expand_dims(real_edge, axis=0)  # add channel dimension
			real_edge = torch.from_numpy(real_edge).float()
			real_edges.append(real_edge.to(device))

			pseudo_image_np = pseudo_optical_detached[i].cpu().numpy().transpose(1, 2, 0)
			pseudo_image_np = (pseudo_image_np * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
			pseudo_edge = cv2.GaussianBlur(pseudo_image_np, (3,3), 0)
			pseudo_edge = cv2.Canny(pseudo_edge, 30, 150)
			pseudo_edge = np.expand_dims(pseudo_edge, axis=0)
			pseudo_edge = torch.from_numpy(pseudo_edge).float()
			pseudo_edges.append(pseudo_edge.to(device))

		true_edges   = torch.stack(real_edges).to(device)
		pseudo_edges = torch.stack(pseudo_edges).to(device)

		# ---------------------
		# Train Discriminator
		# ---------------------
		d_optimizer.zero_grad()

		real_preds  = discriminator(true_optical, true_edges)
		d_real_loss = adversarial_loss(real_preds, torch.ones_like(real_preds))

		fake_preds  = discriminator(pseudo_optical_detached, true_edges)
		d_fake_loss = adversarial_loss(fake_preds, torch.zeros_like(fake_preds))

		d_loss = (d_real_loss + d_fake_loss) / 2
		d_loss.backward()
		d_optimizer.step()

		# ----------------
		# Train Generator
		# ----------------
		g_optimizer.zero_grad()

		fake_preds = discriminator(pseudo_optical, pseudo_edges)
		g_adv_loss = adversarial_loss(fake_preds, torch.ones_like(fake_preds))

		# Reconstruction losses
		rec_mse_loss = mse_loss(pseudo_optical, true_optical)
		rec_ffl = focal_frequency_loss(pseudo_optical, true_optical, rec_weight_matrix)
		rec_vgg = vgg_loss.vgg_loss(pseudo_optical, true_optical)
		rec_style = vgg_loss.style_loss(pseudo_optical, true_optical)

		# Structure Loss
		structure_mse = mse_loss(pseudo_structure_feature, real_structure_feature)

		# Texture Loss
		texture_mse = mse_loss(pseudo_texture_feature, real_texture_feature)
		texture_ffl = focal_frequency_loss(pseudo_texture_feature, real_texture_feature, texture_weight_matrix)

		lambda1 = 10
		lambda2 = 50
		lambda3 = 0.1
		lambda4 = 250
		lambda5 = 0.1
		lambda6 = 1
		lambda7 = 5

		g_loss = (lambda1 * (rec_mse_loss)) + \
				 (lambda2 * (rec_ffl)) + \
				 (lambda3 * rec_vgg) + \
				 (lambda4 * rec_style) + \
				 (lambda5 * g_adv_loss) + \
				 (lambda6 * (structure_mse + texture_mse)) + \
				 (lambda7 * (texture_ffl))

		stabilization_checker.update_losses(g_loss, d_loss)

		# Generate test image after each epoch to see improvement
		with torch.no_grad():
			generator.eval()

			sar_rgb_val = "/ssd/Datasets/sar2opt_raw_data/split_data/test/s1_rgb_tiles/809.png"
			sar_edge_val = "/ssd/Datasets/sar2opt_raw_data/tiles/s1_edge_tiles/809.png"
			sar_gray_val = "/ssd/Datasets/sar2opt_raw_data/split_data/test/s1_gray_tiles/809.png"
			optical_val = "/ssd/Datasets/sar2opt_raw_data/split_data/test/s2_rgb_tiles/809.png"
			sar_rgb_img, sar_edge_img, sar_gray_img, optical_rgb_img = get_validation_input(sar_rgb_val, 
																						    sar_edge_val, 
																						    sar_gray_val, 
																						    optical_val)

			sar_rgb_img      = sar_rgb_img.to(device)
			sar_edge_img     = sar_edge_img.to(device)
			sar_gray_img     = sar_gray_img.to(device)
			optical_rgb_img  = optical_rgb_img.to(device)

			structure_in_val   = torch.cat((sar_gray_img.unsqueeze(0), sar_edge_img.unsqueeze(0)), dim=1)
			structure_mask_val = sar_edge_img.squeeze(1)
			texture_in_val     = sar_rgb_img.unsqueeze(0)
			texture_mask_val   = torch.ones(batch_size, 1, height, width, dtype=sar_rgb_img.dtype, device=sar_rgb_img.device).squeeze(1)

			generated_image, _, _ = generator(texture_in_val, texture_mask_val, structure_in_val, structure_mask_val)

			generated_image = (generated_image + 1) / 2

			generated_image = generated_image.clamp(0, 1)

			save_image(generated_image, f"/ssd/Datasets/sar2opt_raw_data/dualgen_output/epoch_{epoch}.png")
			
		generator.train()

		g_loss.backward()
		g_optimizer.step()

		# Path to save the generator and discriminator state dictionaries
		generator_path = os.path.join(save_dir, f"generator_epoch_{epoch}.pth")
		discriminator_path = os.path.join(save_dir, f"discriminator_epoch_{epoch}.pth")

	# Check if loss has stabilized, adjust LR if it has
	if stabilization_checker.has_stabilized():
		for param_group in d_optimizer.param_groups:
			param_group['lr'] = 0.00005
		for param_group in g_optimizer.param_groups:
			param_group['lr'] = 0.00005 * 10
		freeze_batch_norm(discriminator)
		freeze_batch_norm(generator)
		print(f"Model stabilized at epoch {epoch}.\nAdjusting Learning Rates & Freezing BatchNorm...\n")
	else:
		# Save the state dictionaries
		torch.save(generator.state_dict(), generator_path)
		torch.save(discriminator.state_dict(), discriminator_path)




	# Info log after each output
	epoch_duration = time.time() - epoch_start_time
	print(f"End of Epoch {epoch+1}, Duration: {epoch_duration:.2f} sec")
	print(f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, ")





# def cv2torchRGB(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype("float32") / 255.0
#     img = img.transpose((2, 0, 1))
#     img = torch.tensor(img)
#     return img.unsqueeze(0)

# def cv2torchGRAY(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = img.astype("float32") / 255.0
#     #img = img.transpose((2, 0, 1))
#     img = torch.tensor(img)
#     if img.dim() == 2:
#         img = img.unsqueeze(0).unsqueeze(0)
#     else:
#         img = img.unsqueeze(1)
#     return img

# sar_rgb = cv2.imread("/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_rgb_tiles/16.png")
# sar_gray = cv2.imread("/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_gray_tiles/16.png")
# sar_edge = cv2.imread("/ssd/Datasets/sar2opt_raw_data/split_data/train/s1_edge_tiles/16.png")
# opt_rgb = cv2.imread("/ssd/Datasets/sar2opt_raw_data/split_data/train/s2_rgb_tiles/16.png")

# s_in_gray = cv2torchGRAY(sar_gray)
# s_in_edge = cv2torchGRAY(sar_edge)
# s_in = torch.cat((s_in_gray, s_in_edge), dim=1)
# t_in_rgb = cv2torchRGB(sar_rgb)

# batch_size, channels, height, width = t_in_rgb.size()
# t_in_mask = torch.ones(batch_size, 1, height, width, dtype=t_in_rgb.dtype, device=t_in_rgb.device)

# generator = Generator()
# f = generator(t_in_rgb, t_in_mask.squeeze(1), s_in, s_in_edge.squeeze(1))
# gen_out = f.detach().numpy()[0].T
# discriminator = Discriminator()
# r_probs, fake_probs = discriminator(opt_rgb, gen_out.astype('uint8'))
# print(r_probs)
# print("\n")
# print(fake_probs)
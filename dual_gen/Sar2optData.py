import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize

class SAR2OpticalDataset(Dataset):
	def __init__(self, sar_rgb_dir, sar_edge_dir, sar_gray_dir, optical_rgb_dir, optical_edge_dir, transform=None):
		self.sar_rgb_dir      = sar_rgb_dir
		self.sar_edge_dir     = sar_edge_dir
		self.sar_gray_dir     = sar_gray_dir
		self.optical_rgb_dir  = optical_rgb_dir
		self.optical_edge_dir = optical_edge_dir
		self.transform        = transform

		# list of filenames, files in all directories have same naming convention
		self.image_filenames = [f for f in os.listdir(sar_rgb_dir) if os.path.isfile(os.path.join(sar_rgb_dir, f))]

	def __len__(self):
		return len(self.image_filenames)

	def __getitem__(self, idx):
		sar_rgb_name = os.path.join(self.sar_rgb_dir, self.image_filenames[idx])
		sar_edge_name = os.path.join(self.sar_edge_dir, self.image_filenames[idx])
		sar_gray_name = os.path.join(self.sar_gray_dir, self.image_filenames[idx])
		optical_rgb_name = os.path.join(self.optical_rgb_dir, self.image_filenames[idx])
		optical_edge_name = os.path.join(self.optical_edge_dir, self.image_filenames[idx])

		sar_rgb_img = Image.open(sar_rgb_name).convert('RGB')
		sar_edge_img = Image.open(sar_edge_name).convert('L') # grayscale
		sar_gray_img = Image.open(sar_gray_name).convert('L') # grayscale
		optical_rgb_img = Image.open(optical_rgb_name).convert('RGB')
		optical_edge_img = Image.open(optical_edge_name).convert('L')
		optical_gray_img = Image.open(optical_rgb_name).convert('L')

		transformed_imgs = []

		for img in [sar_rgb_img, sar_edge_img, sar_gray_img, optical_rgb_img, optical_edge_img, optical_gray_img]:
			img = Resize((256, 256))(img)
			img = ToTensor()(img)
			if img.shape[0] == 3:
				img = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
			elif img.shape[0] == 1:
				img = Normalize(mean=[0.5], std=[0.5])(img)
			transformed_imgs.append(img)

		sar_rgb_img      = transformed_imgs[0]
		sar_edge_img     = transformed_imgs[1]
		sar_gray_img     = transformed_imgs[2]
		optical_rgb_img  = transformed_imgs[3]
		optical_edge_img = transformed_imgs[4]
		optical_gray_img = transformed_imgs[5]


		return sar_rgb_img, sar_edge_img, sar_gray_img, optical_rgb_img, optical_edge_img, optical_gray_img


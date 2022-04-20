# collect and save teacher output data set
import argparse
import os
import random
import shutil
import time
import warnings
import math
import pickle
import scipy
import cv2
import numpy as np
import shap
from scipy import ndimage
from enum import Enum
from matplotlib import pyplot as plt
from PIL import Image
from lime.lime_image import LimeImageExplainer
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy import angle, real
from numpy import exp, abs, pi, sqrt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.models as models

from teacher_models import resnet18
from torch.utils.data import Dataset

DATA_PATH = '../datasets/coco/images/'
MODEL_PATH = './model_best-pretrained.pth.tar'
TRAIN_PATH = '../Student/data/train/'
TEST_PATH = '../Student/data/test/'
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SCALE_FACTOR = 8
NUM_EXPLANATIONS = 1
CLASSES = [0, 217, 482, 491, 494, 566, 569, 571, 574, 701]
CLASS_MAP ={0:0, 217:1, 482:2, 491:3, 494:4, 566:5, 569:6, 571:7, 574:8, 701:9} # maps imagenet class target indices to corresponding imagenette ones
NUM_CHANNELS = 3
HEIGHT = 224
WIDTH = 224

class CocoDataSet(Dataset):
	def __init__(self, main_dir, transform):
		self.main_dir = main_dir
		self.transform = transform
		self.total_imgs = os.listdir(main_dir)

	def __len__(self):
		return len(self.total_imgs)

	def __getitem__(self, idx):
		img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
		image = Image.open(img_loc).convert("RGB")
		tensor_image = self.transform(image)
		return tensor_image

# explanations is a list of tuples of the form (explanation_type, img), includes original image
def plot_explanations(explanations, save_path=None):
	fig = plt.figure(figsize=(10, 10))
	cols = 3
	rows = 3

	# ax enables access to manipulate each of subplots
	ax = []

	for i in range(cols*rows):
		if i == NUM_EXPLANATIONS+1:
			break
		explanation_type, img = explanations[i]
		# create subplot and append to ax
		ax.append(fig.add_subplot(rows, cols, i+1))
		ax[-1].set_title(explanation_type)  # set title
		if explanation_type == 'original':
			display = transforms.ToPILImage()(img.squeeze())
			plt.imshow(display)
		elif explanation_type == 'middle layer':
			display = transforms.ToPILImage()(img.squeeze())
			plt.imshow(display, cmap='hot')
		else:
			plt.imshow(img, cmap='hot')

	if save_path is None:
		plt.show()
	else:
		plt.savefig(save_path)

def original(img, visualize=False, save_path=None):
	if visualize:
		display = transforms.ToPILImage()(img.squeeze())
		plt.imshow(display)
		plt.show()
		#plt.savefig(save_path)

def middle_layer_saliency(model, img, transform=None):
	if transform is not None:
		img = transforms.Compose(transform)(img)
	model_output = model(img)
	middle = model_output['middle']
	output = model_output['final']
	middle_explanation = torch.linalg.norm(middle, dim=1, keepdims=True)
	middle_explanation= nn.Upsample(scale_factor=SCALE_FACTOR, mode='bicubic', align_corners=False)(middle_explanation)
	display = transforms.ToPILImage()(middle_explanation.squeeze())
	# plt.imshow(display, cmap='hot')
	# plt.show()

	return middle_explanation, output


def edge_detector(img, transform=None):
	if transform is not None:
		img = transforms.Compose(transform)(img)
	image = img.detach().cpu().numpy().squeeze()
	image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
	grad_x = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
	grad_y = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

	abs_grad_x = cv2.convertScaleAbs(grad_x)
	abs_grad_y = cv2.convertScaleAbs(grad_y)

	grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
	return grad

def saliency_grad(model, img, label, abs_value=True, sum_channel=True, transform=None):
	if transform is not None:
		img = transforms.Compose(transform)(img)
	with torch.enable_grad():
		img.requires_grad = True
		logits = model(img)['final']
		logit = logits[0, label]
		logit.backward()
		grad = img.grad[0].detach().cpu().numpy() # note img.grad is 1x3x224x224, indexes into first dimension

		if abs_value:
			grad = abs(grad)
		if sum_channel:
			grad = grad.sum(axis=0) # sum along first dim (channels)
		return grad

def saliency_smooth_grad(model, img, label, num_repeat=50, noise_level=0.15, abs_value=True, sum_channel=True, transform=None):
	if transform is not None:
		img = transforms.Compose(transform)(img)
	_, _, H, W = img.shape
	device = next(model.parameters()).device
	sigma = noise_level * (img.max() - img.min())

	with torch.enable_grad():
		noises = torch.randn(num_repeat, 3, H, W).to(device) * sigma # 50x3x224x224
		imgs = (torch.cat([img] * num_repeat, dim=0) + noises).to(device) # add noise to image num_repeat times 50x3x224x224
		imgs.requires_grad = True
		logits = model(imgs)['final'] # get prediction logits for noisy images 50x1000
		logit = logits[:, label].sum() # get sum of logits along label column (prediction logits of label class for noisy images)
		logit.backward()
		grads = imgs.grad.detach().cpu().numpy() # 50x3x224x224

		if abs_value:
			grads = abs(grads)
		smooth_grad = grads.mean(axis=0) # 3x224x224
		if sum_channel:
			smooth_grad = smooth_grad.sum(axis=0) # sum along first dim (channels) 224x224

		return smooth_grad

def saliency_gradcam(model, img, label, transform=None):
	if transform is not None:
		img = transforms.Compose(transform)(img)
	_, _, H, W = img.shape
	device = next(model.parameters()).device
	with torch.no_grad():
		conv_maps = model.maxpool(model.relu(model.bn1(model.conv1(img))))
		conv_maps = model.layer4(model.layer3(model.layer2(model.layer1(conv_maps)))).cpu().numpy()[0]
		weights = model.fc.weight.data.cpu().numpy()[label] # get weights of fully connected layer (1000x512) for label class (1x512)
	avg_conv_map = (conv_maps * weights.reshape(-1, 1, 1)).sum(axis=0) # reshape weights to (512x1x1) to multiply with conv_mapså
	avg_conv_map[avg_conv_map < 0] = 0 # sets negative entries to 0
	avg_conv_map = np.array(Image.fromarray(avg_conv_map).resize((H, W)))

	return avg_conv_map

def saliency_lime(model, img, label, transform=None):
	if transform is not None:
		img = transforms.Compose(transform)(img)
	image = transforms.ToPILImage()(img.squeeze())
	assert isinstance(image, Image.Image)
	def predict_prob(imgs):
		with torch.no_grad():
			device = next(model.parameters()).device
			#imgs = [im for im in imgs]
			#imgs = torch.stack(imgs, dim=0)
			imgs_inp = torch.tensor(np.swapaxes(np.swapaxes(imgs,2,3),1,2), dtype=torch.float32).to(device) # convert to desired 200x3x224x224 dims, and float32 dtype
			prob = F.softmax(model(imgs_inp)['final'], dim=-1).cpu().numpy() # model output is 200x1000, softmax is done along 1000 length dim
		return prob
	explainer = LimeImageExplainer()
	image = np.array(image)
	explanation = explainer.explain_instance(image, predict_prob, labels=(label,), hide_color=0, top_labels=None, num_samples=200, batch_size=200)
	coefs = {k: v for k, v in explanation.local_exp[label]}
	attr_map = np.zeros_like(explanation.segments).astype('float')
	for i in range(explanation.segments.max() + 1):
		attr_map[explanation.segments==i] = abs(coefs[i])

	return attr_map

def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
	image = (image - mean) / std

	return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

def saliency_shap(model, img, label, background_dist, abs_value=True, sum_channel=True, transform=None):
	if transform is not None:
		img = transforms.Compose(transform)(img)
	device = next(model.parameters()).device
	class MyNet(nn.Module):
		def __init__(self):
			super().__init__()
			self.model = model
		def forward(self, x):
			logits = self.model(x)['final'][:, label].unsqueeze(1)
			return logits

	with torch.enable_grad():
		explainer = shap.GradientExplainer(MyNet(), normalize(background_dist[0]).to(device), batch_size=128)
		shap_img = img.to(device)
		shap_v = explainer.shap_values(shap_img, rseed=0)
		saliency = shap_v.squeeze()
	if abs_value:
		saliency = abs(saliency)
	if sum_channel:
		saliency = saliency.sum(axis=0)
	# plt.imshow(saliency, cmap='hot')
	# plt.show()
	return saliency

def main():
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
										 std=[1/0.229, 1/0.224, 1/0.225])

	traindir = os.path.join(DATA_PATH, 'train2017')
	train_dataset = CocoDataSet(traindir, transform=transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		])
	)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)


	testdir = os.path.join(DATA_PATH, 'test2017')
	test_dataset = CocoDataSet(testdir, transform=transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		])
	)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

	# normalized_data = datasets.ImageFolder(
	# 	traindir, transforms.Compose([
	# 	transforms.Resize(256),
	# 	transforms.CenterCrop(224),
	# 	transforms.ToTensor(),
	# 	normalize
	# 	])
	# )

	# sample_images = np.zeros((50,224,224,3))
	# sample_targets = np.zeros(50)
	# randomized_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
	# for i, (image, target) in enumerate(randomized_loader):
	# 	if i == sample_images.shape[0]:
	# 		break
	# 	image = np.swapaxes(np.swapaxes(image.squeeze(), 0, 1), 1, 2)
	# 	sample_images[i] = image
	# 	sample_targets[i] = target
	# sample = (np.array(sample_images), np.array(sample_targets))

	standard_transform = [normalize]

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	state = torch.load(MODEL_PATH) # use pre-trained resnet-18 on ImageNet
	model = resnet18()
	model.load_state_dict(state['state_dict'])
	model.to(device)	

	model.eval()
	with torch.no_grad():
		train_labels = []
		test_labels = []
		for i, image in enumerate(train_loader):
			if i % 1000 == 0:
				print(i)
			if i == 10000:
				break
			image = image.cuda(0, non_blocking=True)

			# compute output
			model.eval()
			model_output = model(normalize(image))
			logits = model_output['final']
			logits = logits.detach().cpu().numpy().flatten()
			teacher_label = np.take(logits, CLASSES).argmax() # make sure model is only predicting imagenette classes, not ImageNet

			# make copies for each type of explanation (so no tracing gradients twice)
			images = []
			for _ in range(NUM_EXPLANATIONS):
				images.append(torch.clone(image).to(device))

			#original(image, visualize=True, save_path='original.pdf')
			middle_explanation, output = middle_layer_saliency(model, images[0], transform=standard_transform)
			#plot_middle_frequency(middle_explanation, save_path=freq_path)
			#edges = edge_detector(images[1], transform=standard_transform)
			#grad = saliency_grad(model, images[2], teacher_label, transform=standard_transform)
			#smooth_grad = saliency_smooth_grad(model, images[3], teacher_label, transform=standard_transform)
			#gradcam = saliency_gradcam(model, images[4], teacher_label, transform=standard_transform)
			#lime = saliency_lime(model, images[5], teacher_label, transform=None)
			#shap = saliency_shap(model, images[6], teacher_label, background_dist=sample, transform=standard_transform)
			#explanations = (('original', image.cpu().numpy()), ('middle layer', middle_explanation.cpu().numpy().squeeze()), ('edge detector', edges), ('gradient', grad), ('smoothgrad', smooth_grad), ('gradcam', gradcam), ('lime', lime), ('shap', shap))
			explanations = (('original', image.cpu().numpy()), ('middle layer', middle_explanation.cpu().numpy().squeeze()), ('label', teacher_label))
			training_sample = np.zeros((NUM_CHANNELS + NUM_EXPLANATIONS, HEIGHT, WIDTH))
			training_sample[:NUM_CHANNELS] = image.cpu().numpy().squeeze()
			training_sample[NUM_CHANNELS] = middle_explanation.cpu().numpy().squeeze()
			#plot_explanations(explanations, save_path=save_path)
			save_path = TRAIN_PATH + str(teacher_label) + '/' + str(i) + '.npy'
			np.save(save_path, training_sample)
			train_labels.append(teacher_label)
		for i, image in enumerate(test_loader):
			if i % 1000 == 0:
				print(i)
			if i == 2000:
				break
			image = image.cuda(0, non_blocking=True)

			# compute output
			model.eval()
			model_output = model(normalize(image))
			logits = model_output['final']
			logits = logits.detach().cpu().numpy().flatten()
			teacher_label = np.take(logits, CLASSES).argmax() # make sure model is only predicting imagenette classes, not ImageNet

			# make copies for each type of explanation (so no tracing gradients twice)
			images = []
			for _ in range(NUM_EXPLANATIONS):
				images.append(torch.clone(image).to(device))

			#middle_explanation, output = middle_layer_saliency(model, images[0], transform=standard_transform)
			explanations = (('original', image.cpu().numpy()), ('label', teacher_label))
			save_path = TEST_PATH + str(teacher_label) + '/' + str(i) + '.npy'
			np.save(save_path, image.cpu().numpy().squeeze())
			test_labels.append(teacher_label)

	with open(TRAIN_PATH + 'label_map.pkl', 'wb') as f:
		pickle.dump(train_labels, f)
	with open(TEST_PATH + 'label_map.pkl', 'wb') as f:
		pickle.dump(test_labels, f)
		
if __name__ == '__main__':
	main()
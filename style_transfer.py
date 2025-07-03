import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.optim as optim

imsize = 512
vgg19 = models.vgg19(pretrained=True).features
loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])
for param in vgg19.parameters():
	param.requires_grad = False

class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach()
	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input

class StyleLoss(nn.Module):
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = self.gram_matrix(target_feature).detach()

	def gram_matrix(self, input):
		b, c, h, w = input.size()
		features = input.view(c, h*w)
		G = torch.mm(features, features.t())
		return G.div(c*h*w)

	def forward(self, input):
		G = self.gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input

def get_style_model_and_losses(vgg, style_img, content_img, style_weight = 1000000, content_weight = 1):
	cnn = copy.deepcopy(vgg)

	content_layers = ['conv_4']
	style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

	content_losses = []
	style_losses = []

	model = nn.Sequential()

	i = 0
	for layer in cnn.children():
		if isinstance(layer, nn.Conv2d):
			i += 1
			name = 'conv_{}'.format(i)
		elif isinstance(layer, nn.ReLU):
			name = 'relu_{}'.format(i)
			layer = nn.ReLU(inplace=False)
		elif isinstance(layer, nn.MaxPool2d):
			name = 'pool_{}'.format(i)
		elif isinstance(layer, nn.BatchNorm2d):
			name = 'bn{}'._format(i)
		else:
			raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
		model.add_module(name, layer)

		if name in content_layers:
			target = model(content_img)
			content_loss = ContentLoss(target)
			model.add_module("content_loss_{}".format(i), content_loss)
			content_losses.append(content_loss)
		if name in style_layers:
			target_feature = model(style_img)
			style_loss = StyleLoss(target_feature)
			model.add_module("style_loss_{}".format(i), style_loss)
			style_losses.append(style_loss)
	for i in range(len(model) - 1, -1, -1):
		if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
			break
	model = model[:(i + 1)]

	return model, style_losses, content_losses

def run_style_transfer(vgg, content_img, style_img, input_img, num_steps = 300, style_weight = 1000000, content_weight = 1):
	print('Building the style transfer model..')
	model, style_losses, content_losses = get_style_model_and_losses(vgg, style_img, content_img)
	optimizer = optim.LBFGS([input_img.requires_grad_()])
	print('Optimizing..')
	run = [0]
	while run[0] <= num_steps:
		def closure():
			input_img.data.clamp_(0, 1)
			optimizer.zero_grad()
			model(input_img)
			style_score = 0
			content_score = 0

			for sl in style_losses:
				style_score += sl.loss
			for cl in content_losses:
				content_score += cl.loss

			style_score *= style_weight
			content_score *= content_weight

			loss = style_score + content_score
			loss.backward()

			run[0] += 1
			if run[0] % 50 == 0:
				print("run {}:".format(run))
				print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
				print()
			return style_score + content_score

		optimizer.step(closure)
	input_img.data.clamp_(0, 1)

	return input_img

def image_loader(image_name):
	image = Image.open(image_name)
	image = loader(image).unsqueeze(0)
	return image.to(device, torch.float)

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	style_img = image_loader("style.jpg")
	content_img = image_loader("content.jpg")

	unloader = transforms.ToPILImage()

	input_img = content_img.clone()

	output = run_style_transfer(vgg19, content_img, style_img, input_img)

	output_image = output.cpu().clone().squeeze(0)
	output_image = unloader(output_image)
	output_image.save("output.jpg")


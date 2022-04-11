import torch
import torch.nn as nn
import torch.nn.functional as F 


from .modifiedFaceNet import FaceNet2

class FaceNet(nn.Module):
	"""
	FaceNet
	Zeiler&Fergus
	Require 220x220x3 image
	set margin = 0.2
	SGD
	"""
	def __init__(self, arg):
		super(FaceNet, self).__init__()
		self.arg = arg

		self.conv = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
			nn.BatchNorm2d(64),

			nn.Conv2d(64, 64, kernel_size=(1, 1), stride=1),
			nn.Conv2d(64, 192, kernel_size=(3, 3), stride=1, padding=1),
			nn.BatchNorm2d(192),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

			nn.Conv2d(192, 192, kernel_size=(1, 1), stride=1),
			nn.Conv2d(192, 384, kernel_size=(3, 3), stride=1, padding=1),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

			nn.Conv2d(384, 384, kernel_size=(1, 1), stride=1),
			nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),

			nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),

			nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),

			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),


			nn.Flatten(),
			nn.Linear(4096, 128),
			# nn.ReLU()
			)

	def forward(self, x):
		x = self.conv(x)
		x = F.normalize(x, p=2, dim=1)
		
		return x 



class BasicConv2d(nn.Module):
	"""docstring for BasicConv2d"""
	def __init__(self, inChannels, outChannels, kernel_size, stride, padding=0):
		super(BasicConv2d, self).__init__()

		self.convBatchRelu = nn.Sequential(
			nn.Conv2d(inChannels, outChannels, kernel_size=kernel_size, stride=stride, padding=padding),
			nn.BatchNorm2d(outChannels),
			nn.ReLU(),
			)

	def forward(self, x):
		return self.convBatchRelu(x)
		


class Block35(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(256, 32, kernel_size=1, stride=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
		)

		self.branch2 = nn.Sequential(
			BasicConv2d(256, 32, kernel_size=1, stride=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
		)

		self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		out = torch.cat((x0, x1, x2), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


class Block17(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(896, 128, kernel_size=1, stride=1),
			BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
			BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
		)

		self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


class Block8(nn.Module):

	def __init__(self, scale=1.0, noReLU=False):
		super().__init__()

		self.scale = scale
		self.noReLU = noReLU

		self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(1792, 192, kernel_size=1, stride=1),
			BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
			BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
		)

		self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
		if not self.noReLU:
			self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		if not self.noReLU:
			out = self.relu(out)
		return out


class Mixed_6a(nn.Module):

	def __init__(self):
		super().__init__()

		self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

		self.branch1 = nn.Sequential(
			BasicConv2d(256, 192, kernel_size=1, stride=1),
			BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
			BasicConv2d(192, 256, kernel_size=3, stride=2)
		)

		self.branch2 = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		out = torch.cat((x0, x1, x2), 1)
		return out


class Mixed_7a(nn.Module):

	def __init__(self):
		super().__init__()

		self.branch0 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 384, kernel_size=3, stride=2)
		)

		self.branch1 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 256, kernel_size=3, stride=2)
		)

		self.branch2 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
			BasicConv2d(256, 256, kernel_size=3, stride=2)
		)

		self.branch3 = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		x3 = self.branch3(x)
		out = torch.cat((x0, x1, x2, x3), 1)
		return out


class InceptionResnetV1(nn.Module):
	"""Inception Resnet V1 model with optional loading of pretrained weights.
	Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
	datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
	requested and cached in the torch cache. Subsequent instantiations use the cache rather than
	redownloading.
	Keyword Arguments:
		pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
			(default: {None})
		classify {bool} -- Whether the model should output classification probabilities or feature
			embeddings. (default: {False})
		num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
			equal to that used for the pretrained model, the final linear layer will be randomly
			initialized. (default: {None})
		dropout_prob {float} -- Dropout probability. (default: {0.6})
	"""
	def __init__(self, args):
		super().__init__()
		dropout_prob=0.6

		# Define layers
		self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
		self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
		self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.maxpool_3a = nn.MaxPool2d(3, stride=2)
		self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
		self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
		self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
		self.repeat_1 = nn.Sequential(
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
		)
		self.mixed_6a = Mixed_6a()
		self.repeat_2 = nn.Sequential(
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
		)
		self.mixed_7a = Mixed_7a()
		self.repeat_3 = nn.Sequential(
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
		)
		self.block8 = Block8(noReLU=True)
		self.avgpool_1a = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
		self.dropout = nn.Dropout(dropout_prob)
		self.last_linear = nn.Linear(1792, 512, bias=False)
		# self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)


	def forward(self, x):

		x = self.conv2d_1a(x)
		x = self.conv2d_2a(x)
		x = self.conv2d_2b(x)
		x = self.maxpool_3a(x)
		x = self.conv2d_3b(x)
		x = self.conv2d_4a(x)
		x = self.conv2d_4b(x)
		x = self.repeat_1(x)
		x = self.mixed_6a(x)
		x = self.repeat_2(x)
		x = self.mixed_7a(x)
		x = self.repeat_3(x)
		x = self.block8(x)
		x = self.avgpool_1a(x)
		x = self.dropout(x)
		x = self.last_linear(x.view(x.shape[0], -1))

		# x = self.last_bn(x)
		
		x = F.normalize(x, p=2, dim=1)
		return x


if __name__ == '__main__':
	args = {
	"imageSize": 220
	}
	net = InceptionResnetV1(args)
	a = torch.randn(1, 3, args['imageSize'], args['imageSize'])
	output = net(a)
	print(output.shape)
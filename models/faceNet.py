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
	"""docstring for InceptionResnetV1"""
	def __init__(self, args):
		super(InceptionResnetV1, self).__init__()
		self.args = args
		self.conv = nn.Sequential(
			BasicConv2d(3, 32, kernel_size=3, stride=2),

			BasicConv2d(32, 32, kernel_size=3, stride=1),
			BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),

			nn.MaxPool2d(3, stride=2),
			BasicConv2d(64, 80, kernel_size=1, stride=1),

			BasicConv2d(80, 192, kernel_size=3, stride=1),
			BasicConv2d(192, 256, kernel_size=3, stride=2),

			nn.Sequential(
				Block35(scale=0.17),
				Block35(scale=0.17),
				Block35(scale=0.17),
				Block35(scale=0.17),
				Block35(scale=0.17),
			),

			Mixed_6a(),

			nn.Sequential(
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
			),

			Mixed_7a(),

			nn.Sequential(
				Block8(scale=0.20),
				Block8(scale=0.20),
				Block8(scale=0.20),
				Block8(scale=0.20),
				Block8(scale=0.20),
			),

			Block8(noReLU=True),

			nn.AdaptiveAvgPool2d(1),

			nn.Flatten(),
			nn.Dropout(0.6)
		)
		inFeature = self.conv(torch.randn(1, 3, args['imageSize'], args['imageSize'])).shape[1]
		self.conv.add_module("last_layer", nn.Linear(inFeature, 128))

	def forward(self, x):
		x = self.conv(x)
		x = F.normalize(x)
		return x
		

if __name__ == '__main__':
	args = {
	"imageSize": 220
	}
	net = InceptionResnetV1(args)
	print(net)
	a = torch.randn(1, 3, args['imageSize'], args['imageSize'])
	output = net(a)
	print(output.shape)
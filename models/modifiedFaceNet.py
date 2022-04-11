import torch
import torch.nn as nn
import torch.nn.functional as F 



class BasicConv2d(nn.Module):
	"""docstring for BasicConv2d"""
	def __init__(self, inChannels, outChannels, kernel_size, stride, padding=0):
		super(BasicConv2d, self).__init__()

		self.convBatchRelu = nn.Sequential(
			nn.Conv2d(inChannels, outChannels, kernel_size=kernel_size, stride=stride, padding=padding),
			nn.ReLU(),
			nn.BatchNorm2d(outChannels),
			)

	def forward(self, x):
		return self.convBatchRelu(x)




class FaceNet2(nn.Module):
	"""docstring for FaceNet2"""
	def __init__(self, arg):
		super(FaceNet2, self).__init__()
		self.arg = arg

		self.conv = nn.Sequential(
			BasicConv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

			BasicConv2d(64, 64, kernel_size=(1, 1), stride=1),
			BasicConv2d(64, 192, kernel_size=(3, 3), stride=1, padding=1),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

			BasicConv2d(192, 192, kernel_size=(1, 1), stride=1),
			BasicConv2d(192, 384, kernel_size=(3, 3), stride=1, padding=1),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
			
			BasicConv2d(384, 384, kernel_size=(1, 1), stride=1),
			BasicConv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
			
			BasicConv2d(256, 256, kernel_size=(1, 1), stride=1),
			BasicConv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),

			BasicConv2d(256, 256, kernel_size=(1, 1), stride=1),
			BasicConv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),

			nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

			nn.Flatten(),
			nn.Linear(1024, 128),
			)
	def forward(self, x):
		x = self.conv(x)
		x = F.normalize(x, p=2, dim=1)
		return x
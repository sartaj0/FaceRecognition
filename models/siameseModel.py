import torch
import torch.nn as nn
import torch.nn.functional as F 


class SiameseNetwork(nn.Module):
	"""docstring for SiameseNetwork"""
	def __init__(self, args):
		super(SiameseNetwork, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 64, 10),
			nn.ReLU(inplace=True),
			# nn.BatchNorm2d(64),
			nn.MaxPool2d(2),

			nn.Conv2d(64, 128, 7),
			nn.ReLU(),
			# nn.BatchNorm2d(128),
			nn.MaxPool2d(2),

			nn.Conv2d(128, 128, 4),
			nn.ReLU(),
			# nn.BatchNorm2d(128),
			nn.MaxPool2d(2),

			nn.Conv2d(128, 256, 4),
			nn.ReLU(),

			# nn.MaxPool2d(2),
			# nn.Conv2d(256, 256, 4),
			# nn.ReLU(),

			nn.Flatten()
		)
		outputFeatures = self.conv(torch.randn(1, 3, args['imageSize'], args['imageSize']))

		self.linear = nn.Sequential(nn.Linear(outputFeatures.shape[-1], 1024), nn.ReLU())

	def forward(self, x):
		x = self.conv(x)
		x = self.linear(x)
		return (x)



if __name__ == '__main__':
	args = {
	'imageSize': 128
	}
	# net = SiameseNetwork(imgSize)
	net = FaceNet(args)
	a = torch.randn(1, 3, args['imageSize'], args['imageSize'])
	output = net(a)
	print(output.shape)
	# print(net)
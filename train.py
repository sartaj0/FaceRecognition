import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils import data 
from torch.utils.data.sampler import SubsetRandomSampler


from dataloader import dataset
from models import siameseModel, faceNet


from torchsummary import summary


def save_loss_image(train_loss, val_loss, epoch, model_save_name, model_save_directory):

	fig = plt.figure(figsize=(15, 15))

	plt.plot([k for k in range(1, epoch + 1)], train_loss, label = "Training Loss")
	plt.plot([k for k in range(1, epoch + 1)], val_loss, label = "Validation Loss")
	plt.legend()
	plt.title(model_save_name)
	fig.canvas.draw()
	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	cv2.imwrite(os.path.join(model_save_directory, f"{model_save_name}_loss.jpg"), img)

	plt.close()



def trainTestSplit(args):

	trainDataset = dataset.ImageTriplet(args, train=True)
	valDataset = dataset.ImageTriplet(args, train=True)

	valid_size = 0.2
	num_train = len(trainDataset)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.shuffle(indices)

	train_idx, test_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	test_sampler = SubsetRandomSampler(test_idx) 

	trainDataLoader = data.DataLoader(trainDataset, batch_size=args['batch_size'], shuffle=False, sampler=train_sampler)
	valDataLoader = data.DataLoader(valDataset, batch_size=args['batch_size'], shuffle=False, sampler=test_sampler)

	return trainDataLoader, valDataLoader


def extractLoss(model, criterion, inputs, device):

	anchor = model(inputs[0].to(device))
	# print(anchor.shape)
	# print(anchor.shape)
	positive = model(inputs[1].to(device))
	negative = model(inputs[2].to(device))
	loss = criterion(anchor, positive, negative)
	return loss



def save_model(model, train_loss, val_loss, epoch):
	modelSaveFolder = "save_models"
	if not os.path.isdir(modelSaveFolder):
		os.mkdir(modelSaveFolder)
	if min(val_loss) == val_loss[-1]:
		torch.save(model.state_dict(), os.path.join(modelSaveFolder, "models.pth"))
	save_loss_image(train_loss, val_loss, epoch, "models", modelSaveFolder)



def trainFaceRec(args):

	if args['validationFolder'] is not None:
		trainDataset = dataset.ImageTriplet(args, train=True)
		trainDataLoader = data.DataLoader(trainDataset, batch_size=args['batch_size'], shuffle=True)

		valDataset = dataset.ImageTriplet(args, train=False)
		valDataLoader = data.DataLoader(valDataset, batch_size=args['batch_size'], shuffle=True)

	else:
		trainDataLoader, valDataLoader = trainTestSplit(args)

	# model = siameseModel.SiameseNetwork(args)
	# model = faceNet.FaceNet(args)
	# model = faceNet.FaceNet2(args)
	model = faceNet.InceptionResnetV1(args)
	# print(model)

	criterion = nn.TripletMarginLoss(margin=2.0)
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
	# optimizer = optim.SGD(params=model.parameters(), lr=0.05, momentum=0.9, dampening=0, nesterov=False, weight_decay=1e-5)
	# optimizer = optim.SGD(params=model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-5)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model.to(device)
	summary(model, (3, args['imageSize'], args['imageSize']))

	trainLosses = []
	valLosses = []
	for epoch in range(1, args['epochs'] + 1):
		model.train()
		with tqdm(trainDataLoader, unit=" batch", desc="Training", leave=False) as tepoch:
			trainLoss = []
			for i, inputs in enumerate(tepoch):
				optimizer.zero_grad()

				logits = extractLoss(model, criterion, inputs, device)

				logits.backward()
				optimizer.step()

				loss_value = logits.item()

				tepoch.set_postfix(loss=loss_value)
				trainLoss.append(loss_value)
			train_loss = np.mean(trainLoss)


		model.eval()
		with torch.no_grad():
			with tqdm(valDataLoader, leave=False, desc="Validation") as tepoch:
				valLoss = []
				for i, inputs in enumerate(tepoch):
					logits = extractLoss(model, criterion, inputs, device)
					loss_value = logits.item()
					valLoss.append(loss_value)
					tepoch.set_postfix(loss=loss_value)

			val_loss = np.mean(valLoss)

		print(f"Epochs {epoch}\t Training Loss: {train_loss}\t Testing Loss: {val_loss}")
		trainLosses.append(train_loss)
		valLosses.append(val_loss)

		save_model(model, trainLosses, valLosses, epoch)


	model.load_state_dict(torch.load(os.path.join("save_models", "models.pth")))
	model.to("cpu")
	model.eval()

	dummy_input = torch.randn(1, 3, args['imageSize'], args['imageSize'])
	torch.onnx.export(model, dummy_input, 
		os.path.join("save_models", "models.onnx"), verbose=True)


if __name__ == '__main__':
	args={
	"validationFolder": r"E:\dataset\Face\Bolly\Faces",
	"trainFolder": r"E:\dataset\Face\lfw_224",
	# "validationFolder": None,
	"imageSize": 128,
	"epochs": 100,
	'batch_size': 32,
	"rgb": True,
	'fixedPairs': False,
	}

	trainFaceRec(args)
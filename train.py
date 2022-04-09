import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils import data 
from torch.utils.data.sampler import SubsetRandomSampler


import Dataset
from models import siameseModel, faceNet


def trainTestSplit(args):

	trainDataset = Dataset.ImageTriplet(args, train=True)
	valDataset = Dataset.ImageTriplet(args, train=True)

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
	positive = model(inputs[1].to(device))
	negative = model(inputs[2].to(device))
	loss = criterion(anchor, positive, negative)
	return loss



def trainFaceRec(args):

	if args['validationFolder'] is not None:
		trainDataset = Dataset.ImageTriplet(args, train=True)
		trainDataLoader = data.DataLoader(trainDataset, batch_size=args['batch_size'], shuffle=True)

		valDataset = Dataset.ImageTriplet(args, train=False)
		valDataLoader = data.DataLoader(valDataset, batch_size=args['batch_size'], shuffle=True)

	else:
		trainDataLoader, valDataLoader = trainTestSplit(args)

	# model = siameseModel.SiameseNetwork(args)
	# model = faceNet.FaceNet(args)
	model = faceNet.InceptionResnetV1(args)
	print(model)

	criterion = nn.TripletMarginLoss(margin=0.2)
	optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
	# optimizer = optim.SGD(params=model.parameters(), lr=0.05, momentum=0.9, dampening=0, nesterov=False, weight_decay=1e-5)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model.to(device)

	trainLosses = []
	valLosses = []
	for epoch in range(1, args['epochs'] + 1):
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


if __name__ == '__main__':
	args={
	"trainFolder": r"E:\dataset\Face\dataset3\train",
	"validationFolder": r"E:\dataset\Face\dataset3\val",
	# "validationFolder": None,
	"imageSize": 220,
	"epochs": 100,
	'batch_size': 8,
	"rgb": True
	}

	trainFaceRec(args)
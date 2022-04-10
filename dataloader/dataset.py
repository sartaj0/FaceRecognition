from torch.utils import data 
from torchvision import transforms as T

import os
import cv2
import random
from PIL import Image


def generateFixedTriple(folder):
	triplets = {}
	persons = {}
	
	j = 0
	for i, person in enumerate(os.listdir(folder)):
		personFolder = os.path.join(folder, person)
		personFolderImages = [os.path.join(personFolder, imagename) for imagename in os.listdir(personFolder)]
		persons[i] = personFolderImages

	personsID = persons.keys()

	for i, person in persons.items():
		for imagename in person:
			anchor = imagename 
			positive = random.choice(tuple(set(person) - set([imagename])))
			negative = random.choice(persons[random.choice(tuple(set(personsID) - set([i])))])
			
			triplets[j] = (anchor, positive, negative)

			j += 1
	return triplets


def generateRandomTriple(idxsValue, persons, idx):
	anchor, person = idxsValue[idx]
	
	positive = random.choice(tuple(set(persons[person]) - set([anchor])))
	person2 = random.choice(tuple(set(persons.keys()) - set([person])))
	negative = random.choice(persons[person2])

	return anchor, positive, negative



def generateRandom(mainFolder):

	j = 0
	idxsValue = {}
	persons = {}
	for i, person in enumerate(os.listdir(mainFolder)):
		personFolder = os.path.join(mainFolder, person)

		imgPaths = [os.path.join(personFolder, img) for img in os.listdir(personFolder)]

		if len(imgPaths) < 2:
			continue
		persons[person] = []
		for imgPath in imgPaths:
			idxsValue[j] = [imgPath, person]
			persons[person].append(imgPath)
			j += 1
	return idxsValue, persons




def transformations(args):
	if args['rgb']:
		transforms = T.Compose([
			T.Resize((args['imageSize'], args['imageSize'])),
			T.ToTensor(),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	else:
		transforms = T.Compose([
			T.Resize((args['imageSize'], args['imageSize'])),
			T.Grayscale(1),
			T.ToTensor(),
			T.Normalize((0.5, ), (0.5, ))])

	return transforms



class ImageTriplet(data.Dataset):
	"""docstring for ImageTriplet"""
	def __init__(self, args, train):
		self.args = args
		super(ImageTriplet, self).__init__()

		if train:
			args['folder'] = args['trainFolder']
		else:
			args['folder'] = args['validationFolder']


		if args['fixedPairs']:
			self.triplets = generateFixedTriple(args['folder'])
			self.len = len(self.triplets)


		else:
			self.idxsValue, self.persons = generateRandom(args['folder'])

			self.len = len(self.idxsValue)

		self.transform = transformations(args)

	def __len__(self):
		return self.len


	def __getitem__(self, idx):

		if self.args['fixedPairs']:
			anchor = self.readImage(self.triplets[idx][0])
			positive = self.readImage(self.triplets[idx][1])
			negative = self.readImage(self.triplets[idx][2])

		else:
			anchor, positive, negative = generateRandomTriple(self.idxsValue, self.persons, idx)

			anchor = self.readImage(anchor)
			positive = self.readImage(positive)
			negative = self.readImage(negative)

		return anchor, positive, negative


	def readImage(self, imagePath):
		image = Image.open(imagePath).convert('RGB')
		image = self.transform(image)
		return image 


def viewInput(tensorTuple):
	images = []
	for image in tensorTuple:
		image = image.permute(1, 2, 0)
		image = image.detach().numpy()
		# print(image.max(), image.min())

		image *= 0.5
		image += 0.5
		# image = image.reshape(128, 128, 3)
		images.append(image)

	image = cv2.hconcat(images)
	cv2.imshow("image", image)
	cv2.waitKey(0)

if __name__ == '__main__':
	args = {
	"trainFolder": r"E:\dataset\Face\dataset",
	"imageSize": 128,
	"rgb": True,
	"fixedPairs": False,
	}
	dataset = ImageTriplet(args, train=True)
	print(len(dataset))
	viewInput(dataset[0])
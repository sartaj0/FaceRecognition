import os 
import cv2
import numpy as np 


def extract_face(net, img, returnCoords=False, confidence=0.5):
	x1, y1, x2, y2 = detect_faces(net, img.copy(), returnCoords=False, confidence=0.5)[0]
	return img[y1: y2, x1: x2]

def detect_faces(net, img, returnCoords=False, confidence=0.5):
	
	height, width = img.shape[:2]

	blob = cv2.dnn.blobFromImage(image=cv2.resize(img, (300, 300)), 
		scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=True)

	net.setInput(blob)
	detection = net.forward()

	face_rect = detection[detection[:,:,:,2] > confidence]
	boxes = face_rect[:, 3:7] * np.array([width, height, width, height])
	boxes = boxes.astype(int)

	coords = []
	for box in boxes:
		x1, y1, x2, y2 = box
		coords.append([x1, y1, x2, y2])

	return coords


def detect_vector(net, face, imgSize):
	face = cv2.cvtColor(cv2.resize(face, (imgSize, imgSize)), cv2.COLOR_BGR2RGB) / 255.0
	face -= [0.5, 0.5, 0.5]
	face /= 0.5
	facenet.setInput(face.reshape(1, 3, imgSize, imgSize))
	vector = facenet.forward()
	print(f"Vector: {vector.shape} {vector.max()} {vector.min()}")

	return vector



def compare_vector(vector1, vector2):
	pass

if __name__ == '__main__':
	prototxt_loc = r"E:\Models\DeepLearning\FaceDetection\deploy.prototxt.txt"
	model_loc = r"E:\Models\DeepLearning\FaceDetection\res10_300x300_ssd_iter_140000.caffemodel"
	faceDetector = cv2.dnn.readNetFromCaffe(prototxt_loc, model_loc)

	imgSize = 128

	facenet = cv2.dnn.readNetFromONNX("save_models/models.onnx")

	img1 = cv2.imread(r"E:\Projects\FaceRecognition\test_images\1.jpg")
	face1 = extract_face(faceDetector, img1)
	vector1 = detect_vector(facenet, face1, imgSize)

	img2 = cv2.imread(r"E:\Projects\FaceRecognition\test_images\2.jpg")
	face2 = extract_face(faceDetector, img2)
	vector2 = detect_vector(facenet, face2, imgSize)


	img3 = cv2.imread(r"E:\Projects\FaceRecognition\test_images\3.jpg")
	face3 = extract_face(faceDetector, img3)
	vector3 = detect_vector(facenet, face3, imgSize)

	cv2.imshow("face1", face1)
	cv2.imshow("face2", face2)
	cv2.imshow("face3", face3)

	
	difference1 = np.sum(np.power(vector1 - vector2, 2))
	difference2 = np.sum(np.power(vector1 - vector3, 2))
	print(difference1, difference2)
	print(max(difference1 - difference2 + 0.2, 0))
	# print(max(difference1 - difference2 , 0))
	# print(max(difference2 - difference1 , 0))

	# print(difference1, difference2)
	cv2.waitKey(0)



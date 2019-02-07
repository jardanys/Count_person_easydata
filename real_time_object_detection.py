# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")

args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["FONDO", "AVION", "BICICLETA", "PAJARO", "BARCO",
	"BOTELLA", "BUS", "AUTO", "GATO", "SILLA", "VACA", "PISO",
	"DOG", "CABALLO", "MOTO", "PERSONA", "PLANTA", "OVEJA",
	"SOFA", "TREN", "TV"]
# CLASSES = ["cat", "dog", "person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
writer = None

ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#vs = cv2.VideoCapture('videoprueba3.mp4')
#vs = cv2.VideoCapture('http://cgi-bin/snapshot.cgi?chn=1&u=admin&p=admin2018')
#vs = cv2.VideoCapture('rtsp://admin:8046055EDS@192.168.0.3:554/cam/realmonitor?channel=10&subtype=1')
#vs = cv2.VideoCapture('rtsp://admin:admin2018@192.168.1.254:554/streaming/channels/101')
vs = cv2.VideoCapture('rtsp://admin:admin2018@192.168.1.53:554/streaming/channels/301')
#vs = cv2.VideoCapture('rtsp://admin:admin2018@255.255.255.254:554/streaming/channels/101')
#vs = cv2.VideoCapture('rtsp://admin:admin2018@190.157.8.33:554/streaming/channels/101')
#vs = cv2.VideoCapture('rtsp://admin:admin2018@190.157.8.1:554/streaming/channels/101')
#vs = cv2.VideoCapture('rtsp://admin:admin2018@192.0.0.64:554/streaming/channels/101')
#vs = cv2.VideoCapture('rtsp://admin:admin2018@100.79.20.118:554/streaming/channels/101')
#vs = cv2.VideoCapture('rtsp://admin:admin2018@192.168.1.254:554/cam/realmonitor?channel=10&subtype=1')
#vs = cv2.VideoCapture('rtsp://192.168.0.3:554/cam/realmonitor?channel=4&subtype=0&unicast=true&proto=Onvif')
#rtsp://admin:admin@10.7.6.67:554/cam/realmonitor?channel=1&subtype=1

time.sleep(4.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the
	# end of the stream
	if not grabbed:
		break

	#frame = vs.read()
	frame = imutils.resize(frame, width=650)

	# grab the frame dimensions and convert it to a blob
	(H, W) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)),
		0.007843, (400, 400), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			rects.append(box.astype("int"))

			# draw the prediction on the frame
			label = "{}".format(CLASSES[idx])
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			#print(idx, "DOG CLASSIFICATION")

#	objects = ct.update(rects)
#	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
#		text = "PET {}".format(objectID)
#		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
#			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"WMV2")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(640, 480))

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
if writer is not None:
	writer.release()

# real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --output webcam_face_recognition_output.avi

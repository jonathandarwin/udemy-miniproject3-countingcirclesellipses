import cv2
import numpy as np

# gambar yang ingin dideteksi blob nya harus di grayscale dlu
image = cv2.imread('blobs.jpg', 0)
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# create detector
# blob : group pixel gambar yang miliki property yang sama, seperti warna, intensitas, dll.
detector = cv2.SimpleBlobDetector_create()

# detect blobs
kp = detector.detect(image)

# draw blobs
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, kp, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

numberOfBlobs = len(kp)
text = 'Total number of blobs : ' + str(numberOfBlobs)
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

# display image with blob keypoints
cv2.imshow('Blob using default parameters' ,blobs)
cv2.waitKey(0)

# kita dapat set parameter ketika menggunakan SimpleBlobDetector 
# set parameter yang sesuai dengan blob yang ingin kita detect
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100

params.filterByCircularity = True
params.minCircularity = 0.9

params.filterByConvexity = False
params.minConvexity = 0.2

params.filterByInertia = True
params.minInertiaRatio = 0.01

# create detector
detector = cv2.SimpleBlobDetector_create(params)

# detect blobs
kp = detector.detect(image)

# draw blobs
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, kp, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

numberOfBlobs = len(kp)
text = 'Number of circular blobs : ' + str(numberOfBlobs)
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

cv2.imshow('Filtering Circular Blobs Only', blobs)
cv2.waitKey(0)
cv2.destrotAllWindows()
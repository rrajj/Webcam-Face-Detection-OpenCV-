import cv2

class FaceDetector:

    def __init__(self, faceCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)


    def detectFaces(self, image, scaleFactor = 1.5, minNeighbors = 5, minSize = (30, 30)):
        rectangles = self.faceCascade.detectMultiScale(image, scaleFactor = scaleFactor,
                                                       minNeighbors = minNeighbors, minSize = minSize, flags = cv2.CASCADE_SCALE_IMAGE)
        return rectangles
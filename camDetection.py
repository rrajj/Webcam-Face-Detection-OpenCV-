import argparse
import cv2
import faceDetector

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True, help = "Face Cascade Location")
args = vars(ap.parse_args())

fd = faceDetector.FaceDetector(args["face"])
camera = cv2.VideoCapture(0)    #use the webcam

#now to start looping on all the frames

while True:
    (grabbed, frame) = camera.read()

    # to convert the collected frame to grayscale
    # if need the rgb one 'uncomment' the below line
    # and replace 'gray' with 'frame' where necessary
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not grabbed:     #in case user permanently stops the execution of the script
        break

    faceRects = fd.detectFaces(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))

    frameClone = gray.copy()       #create a clone of existing frame for just in case processing

    for (X, Y, width, height) in faceRects:
        cv2.rectangle(frameClone, (X, Y), (X + width, Y + height), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frameClone)


    #below mentioned line makes sure user hits "q" in order to quit.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
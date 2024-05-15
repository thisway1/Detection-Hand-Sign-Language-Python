import cv2  # Lib OpenCV
from cvzone.HandTrackingModule import HandDetector  # CVZONE MEDIAPIPE
from cvzone.ClassificationModule import Classifier  # CVZONE TENSORFLOW
import numpy as np
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Hand detection setup
detector = HandDetector(maxHands=1)
offset = 20
sizeImg = 300

# Load the classifier model and labels
classifier = Classifier("D:\skripsi\pythonProject4\Model\keras_model.h5", "D:\skripsi\pythonProject4\Model\labels.txt")
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Path for saving capturing images
fldr = "D:\skripsi\pythonProject4\Dat\A"
countr = 0

while True:
    success, img = cap.read()  # Get frame from webcam
    imgOutpt = img.copy()  # Create a copy of the frame
    hand, img = detector.findHands(img)  # Detect hands in the frame

    if hand:
        hnd = hand[0]
        x, y, w, h = hnd['bbox']

        # Create a white background image
        imgWhite = np.ones((sizeImg, sizeImg, 3), np.uint8) * 255

        # Crop and resize the hand
        Imgcrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        aspectRatio = h / w
        # Fix Height and Width
        if aspectRatio > 1:
            # Fix Height
            k = sizeImg / h
            widthCal = math.ceil(k * w)
            Resize = cv2.resize(Imgcrop, (widthCal, sizeImg))
            # Find Gap for centering
            wiGap = math.ceil((sizeImg - widthCal) / 2)
            imgWhite[:, wiGap:widthCal + wiGap] = Resize

            # Get classification prediction and index
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)
        else:
            # Fix Width
            k = sizeImg / w
            heightCal = math.ceil(k * h)
            Resize = cv2.resize(Imgcrop, (sizeImg, heightCal))
            # Find Gap for centering
            hiGap = math.ceil((sizeImg - heightCal) / 2)
            imgWhite[hiGap:heightCal + hiGap, :] = Resize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Display cropped hand and white screen images
        cv2.imshow("Crop", Imgcrop)
        cv2.imshow("white screen", imgWhite)



        # Display result classification label on the main frame
        cv2.putText(imgOutpt, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 1)
        # Make a Rectangle in object
        cv2.rectangle(imgOutpt, (x-offset, y-offset), (x+w+offset, y+h+offset), (10, 43, 255), 1)

    # Display the main frame with detected hand and classification label
    cv2.imshow("Images", imgOutpt)
    key = cv2.waitKey(1)

    # Save images if 's' key is pressed
    if key == ord("s"):
        countr += 1
        cv2.imwrite(f'{fldr}/Image_{time.time()}.jpg', imgWhite)
        print(countr)

    # Exit the loop when the 'q' key is pressed
    if key == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

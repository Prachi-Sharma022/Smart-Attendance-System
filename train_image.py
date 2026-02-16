import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []

    # Load the haarcascade detector
    haarcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(haarcascadePath)

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')  # convert to grayscale
        imageNp = np.array(pilImage, 'uint8')

        # Detect faces in the image
        facesDetected = detector.detectMultiScale(imageNp)
        Id = int(os.path.split(imagePath)[-1].split(".")[1])  # filename must be User.ID.count.jpg

        for (x, y, w, h) in facesDetected:
            faces.append(imageNp[y:y+h, x:x+w])  # crop face
            Ids.append(Id)

    return faces, Ids



def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    haarcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(haarcascadePath)
    faces, Id = getImagesAndLabels("ImageBasic")
    recognizer.train(faces, np.array(Id))
    Thread(target=counter_img, args=("ImageBasic",)).start()
    if not os.path.exists("TrainingImageLabel"):
        os.makedirs("TrainingImageLabel")
    recognizer.save("TrainingImageLabel/Trainer.yml")
    print("Training Completed")



def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1

if __name__ == "__main__":
    TrainImages()
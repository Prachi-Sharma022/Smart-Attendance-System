import csv
import cv2
import os


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False



def takeImages():
    Id = input("Enter Your Id: ")
    name = input("Enter Your Name: ")

    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        dataset_folder = "ImageBasic"
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
                sampleNum += 1
                cv2.imwrite(os.path.join(dataset_folder, f"{name}.{Id}.{sampleNum}.jpg"), gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()

        # Save ID and Name in CSV
        folder = "StudentDetails"
        if not os.path.exists(folder):
             os.makedirs(folder)
        row = [Id, name]
        with open(os.path.join(folder, "StudentDetails.csv"), 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

        print(f"Images Saved for ID : {Id} Name : {name}")

    else:
        if not is_number(Id):
            print("Enter Numeric ID")
        if not name.isalpha():
            print("Enter Alphabetical Name")

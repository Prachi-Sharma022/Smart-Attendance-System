import datetime
import os
import time
import cv2
import pandas as pd


def recognize_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  
    recognizer.read("TrainingImageLabel"+os.sep+"Trainer.yml")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) 
    cam.set(4, 480) 
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,
                                             minSize=(int(minW), int(minH)),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            confidence = 100 - conf


            THRESHOLD = 65
            if confidence >= THRESHOLD:
                aa = df.loc[df['Id'] == Id]['Name'].values
                name_str = str(aa[0]) if len(aa) > 0 else "Unknown"
                attendance.loc[len(attendance)] = [Id, name_str,
                                                    datetime.datetime.now().strftime('%Y-%m-%d'),
                                                    datetime.datetime.now().strftime('%H:%M:%S')]
                tt = f"{Id}-{name_str} [Pass]"
            else:
                tt = "Unknown"

            cv2.putText(im, str(tt), (x+5, y-5), font, 1, (255, 255, 255), 2)
            if confidence > THRESHOLD:
                color = (0, 255, 0)
            elif confidence > 50:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.putText(im, f"{round(confidence)}%", (x+5, y+h-5), font, 1, color, 1)

        # After the loop, before saving
        attendance.drop_duplicates(subset=['Id'], keep='first', inplace=True)

        cv2.imshow('Attendance', im)
        if cv2.waitKey(1) == ord('q'):
            break

    if not os.path.exists("Attendance"):
        os.makedirs("Attendance")
    ts = time.time()
    date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    fileName = f"Attendance{os.sep}Attendance_{date_str}.csv"
    attendance.to_csv(fileName, index=False)
    print("Attendance Successful")
    cam.release()
    cv2.destroyAllWindows()

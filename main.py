# install dlib
# install os
# install face_recognition
# install numpy
# install cmake


import cv2
import numpy as np
import face_recognition
import os


path = 'imageFolder'
images = []
className = []
mylist = os.listdir(path)
print(mylist)
#f = open("customer_arry.txt","w")
#x = f.write(mylist)
#f.close()

for cls in mylist:
    img = cv2.imread(f'{path}/{cls}')
    images.append(img)
    className.append(os.path.splitext(cls)[0])
print(className)


def findEncodings(images):
    encodeList = []
    for im in images:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(im)[0]
        encodeList.append(encode)
    return encodeList


encodingListKnown = findEncodings(images)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    s, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurLocFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurLocFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurLocFrame):
        matches = face_recognition.compare_faces(encodingListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodingListKnown, encodeFace)
        print(faceDis)
        print(matches)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            print(faceLoc)
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (250, 0, 230), 2)
            cv2.rectangle(img, (x1, (y2-35)), (x2, y2), (255, 0, 150), cv2.FILLED)
            cv2.putText(img, name, ((x1+6), (y2-6)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            y1, x2, y2, x1 = faceLoc
            print(faceLoc)
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (250, 0, 230), 2)
            cv2.rectangle(img, (x1, (y2 - 35)), (x2, y2), (255, 0, 150), cv2.FILLED)
            cv2.putText(img, "unkhown", ((x1 + 6), (y2 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('WEBCAM', img)
    cv2.waitKey(1)



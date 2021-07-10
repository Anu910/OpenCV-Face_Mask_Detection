import cv2
import numpy as np
import matplotlib as plt

capture=cv2.VideoCapture(0)
data=[]

faceCascade=cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
while True:
    flag,img=capture.read()
    if flag:
        faces= faceCascade.detectMultiScale(img)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<300:
                data.append(face)
        
        cv2.imshow("output",img)
    if cv2.waitKey(1) & 0xFF ==ord('q') or len(data)>=200:
        break
capture.release()

cv2.destroyAllWindows()

np.save("with_mask",data)


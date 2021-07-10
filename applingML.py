import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#to reduce the dimensionality of the data
#PCA--principal componenet analysis
from sklearn.decomposition import PCA


#loading mask and no mask datasets

with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')

#reshape the datasets(contagiously aranging(not exact but similar))
with_mask=with_mask.reshape(200,50*50*3)

without_mask=without_mask.reshape(200,50*50*3)

#x coordinate --concatinating both with_mask and without_mask
X=np.r_[with_mask,without_mask]

print(X.shape)

#(400,7500)

#Y coordinate
labels=np.zeros(X.shape[0])

# lbleing with mask images as 1.0 and remainig 0-199 are 0.0 default
labels[200:]=1.0

names={0:'Mask',1:'No Mask'}

#here we are performing binary classification on labeled data---supervised machinelearning

x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.25)

#reducing the dimensionality of our data
#we use eigen values and eigen vectors in dimensionality reduction
#7500 is reduced to 3
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)

#print(x_train[0])

#print(x_train.shape)

#ML
svm=SVC()
svm.fit(x_train,y_train)

x_test=pca.transform(x_test)
y_pred=svm.predict(x_test)

#accuracy 1.0 is 100% this means our model overfits the data 
# so we do shuffling in the training and testing data or by putting value of test_size in different way

accuracy=accuracy_score(y_test,y_pred)

#we have model ready and we now start our camera to capture and test
faceCascade=cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

capture=cv2.VideoCapture(0)
while True:
    flag,img=capture.read()
    if flag:
        faces= faceCascade.detectMultiScale(img)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face=pca.transform(face)
            #prediction will be a 1 or 0
            pred=svm.predict(face)
            n=names[int(pred)]
            cv2.putText(img,n,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            print(n)
        cv2.imshow("output",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
capture.release()

cv2.destroyAllWindows()


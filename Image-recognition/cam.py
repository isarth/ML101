import cv2
from matplotlib import pyplot 
import numpy as np
# import numpy as np
f=np.load('./data/my_face_data.npy')
f = f.reshape((f.shape[0],f.shape[1]*f.shape[2]))
labels=np.load('./data/my_face_labels.npy')
def dist(x1,x2):
    #print x1.shape,x2.shape
    return np.sqrt(sum((x1-x2)**2))
def knn(X_train,y_train,X_test,k=2):
    n_train=X_train.shape[0]
    #n_test=X_test.shape[0]
    val=[]

    distance=[]
    for iy in range(n_train):
            distance.append([dist(X_train[iy],X_test),y_train[iy]])
    distance.sort()
    distance=np.array(distance[:k])
    unique,neighbours=np.unique(distance[:,1],return_counts=True)
    ans=dict(zip(unique,neighbours))
    sorted_ans=sorted(ans,key=lambda x:ans[x])
    print sorted_ans[-1]
    return  sorted_ans[-1]    
rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# print facec
font = cv2.FONT_HERSHEY_SIMPLEX


def get_name(im):
    return knn(f[:,:],labels[:],im)
def recognize_face(im):
    im = cv2.resize(im, (100, 100))
    im = im.flatten()

    return get_name(im)

while True:
    _, fr = rgb.read()
    #gray2=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # print gray.shape
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        gray2 = gray[x:x+w, y:y+h]
        print gray2.shape
        if gray2.shape[0]==0:
            break
        out = recognize_face(gray2)
        cv2.putText(gray, out, (x, y), font, 1, (255, 255, 0), 2)
    	cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    
    #cv2.imshow('rgb', fr)
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

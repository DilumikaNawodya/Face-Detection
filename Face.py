import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

DATADIR = 'Your datadirectory goes'
CATEGORY = ["To be","Done"]

opt = int(input("Enter 1 to predict on a image or 0 to predict on live feed: ")) 
i = 1
if opt==1:
    open_path = os.path.join(DATADIR,CATEGORY[0])
    save_dir = os.path.join(DATADIR,CATEGORY[1])
    for img in os.listdir(open_path):
        
        frame = cv2.imread(os.path.join(open_path,img))
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        
        for (x,y,w,h) in faces:
            
            face = gray[y:y+h,x:x+w]
            resized_image = cv2.resize(face,(50,50))
            normalized_image = resized_image/255.0
            
            reshaped_image = np.reshape(normalized_image,(1,50,50,1))
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imwrite(os.path.join(save_dir, str(i)+".jpg"), frame)
        i += 1


else:
    video_capture = cv2.VideoCapture(0)
    
    while True:
        _,frame = video_capture.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        
        for (x,y,w,h) in faces:
            
            face = gray[y:y+h,x:x+w]
            resized_image = cv2.resize(face,(50,50))
            normalized_image = resized_image/255.0
            
            reshaped_image = np.reshape(normalized_image,(1,50,50,1))

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
        cv2.imshow('Video',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
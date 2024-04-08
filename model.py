#import face_recognition
import os
import cv2 as cv
import numpy as np
import pandas as pd
import time
from datetime import datetime
from keras.models import load_model
from keras import layers
import tensorflow
import keras
from tensorflow.keras import backend
from ultralytics import YOLO
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance
    between the embeddings
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, compare):
        sum_squared = backend.sum(backend.square(anchor - compare), axis=1, keepdims=True)
        return backend.sqrt(backend.maximum(sum_squared, backend.epsilon()))
custom_object = {
    'DistanceLayer': DistanceLayer ,
}




recognition_model=load_model("models\Simease_model.keras",custom_objects=custom_object)
yolo_face_model=YOLO("D:\projects\Face_recognition\models\yolo_face_extract.pt")


class Register:
    def __init__(self):
        self.encodingArr = np.load("utilsFiles\\encodingArr.npy")
        self.labelArr = np.load("utilsFiles\\labelArr.npy")
        self.StdData = pd.read_csv("utilsFiles\StdDetails.csv")
        self.columns = self.StdData.columns
        self.imageLis=[]

        

    def checkPresence(self, inputArr):
        
        for i, arr in enumerate(self.encodingArr):
            
            #result = face_recognition.compare_faces(arr, inputArr)
            arr=np.reshape(arr,(1,224,224,3))
            inputArr=np.reshape(inputArr,(1,224,224,3))
            result=recognition_model.predict([arr,inputArr])[0]
            print("Result:",result)
            if result>80.0:
               
                return (True, i+1)
        

        return (False, -1)
        

    def addPerson(self, inputArr, Name):
        exist, index = self.checkPresence(inputArr)
       
        if not exist:
            self.imageLis.append(inputArr)
            self.encodingArr = np.array( self.imageLis)
            self.labelArr = np.append(self.labelArr, [self.labelArr.shape[0]], axis=0)
            
            data = pd.DataFrame({self.columns[0]: [Name], self.columns[1]: [len(self.labelArr)]})
            self.StdData = pd.concat([self.StdData, data], ignore_index=True)  # Use pd.concat instead of append
            self.saveToData()
            print("Registration Completed")

        else:
            name = self.StdData[self.StdData['Index'] == index].values[0][0]
            print(f"{name} Already Exist in the Database")
      
    def saveToData(self):
        self.StdData.to_csv("utilsFiles\StdDetails.csv", index=False)
        np.save("utilsFiles\\encodingArr.npy", self.encodingArr)
        np.save("utilsFiles\\labelArr.npy", self.labelArr)
        print("Saved Sucessfully")



class Recognize(Register):
    def __init__(self) :
        super().__init__()

    def start_face_recognition(self):
        name="unknown"
        data=pd.read_csv("utilsFiles/StdDetails.csv")
        video_capture = cv.VideoCapture(0)

        face_locations = []
        face_encodings = []
        face_names = []
        
        while True:


            ret, frame = video_capture.read()
            
            start=time.time()
            # face_locations = face_recognition.face_locations(frame)
            # face_encodings = face_recognition.face_encodings(frame)
            results=yolo_face_model(frame)
            box=results[0].boxes[0]
            
            top_left_x=int(box.xyxy.tolist()[0][0])
            top_left_y=int(box.xyxy.tolist()[0][1])
            bottom_right_x=int(box.xyxy.tolist()[0][2])
            bottom_right_y=int(box.xyxy.tolist()[0][3])
            extracted_image = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            new_image=cv.resize(extracted_image,(224,224))

            face_encoding=new_image
            find,i=super().checkPresence(face_encoding)
            if find:
                
                name=data[data['Index']==i].values[0][0]
                print(f"Person {name} detected")
                #face_names.append(name)
 
            cv.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

            font = cv.FONT_HERSHEY_DUPLEX
                
            print(f"Name {name} and type {type(name)}")
            cv.putText(frame, name, (top_left_x + 6, bottom_right_x - 6), font, 1.0, (255, 255, 255), 1)
            end=time.time()
            cv.imshow('Video', frame)
            time1=end-start
            print(f"Total Time{time1}")
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


        video_capture.release()
        cv.destroyAllWindows()

    def capture_good_quality_frame(self):
        cap = cv.VideoCapture(0)

        # Set camera parameters for better quality
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)  
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080) 
        cap.set(cv.CAP_PROP_FPS, 30)  

    
        for _ in range(30):
            _, _ = cap.read()

        while cap.isOpened():
            find, frame = cap.read()

            

            cv.imshow("Frame", frame)

            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    
        cap.release()
        cv.destroyAllWindows()

        return frame
    

class Start(Recognize):
    def __init__(self):
        super().__init__()

    def launch(self):
        while True:

            x = int(input("Press 1 to register\nPress 2 to start Verification\nPress -1 to exit"))
            if x == 1:
                name = input("Enter your name")
                verified = 0
                encoding=0
                while True:
                    frame = super(Start, self).capture_good_quality_frame() 
                    results=yolo_face_model(frame)
                    box=results[0].boxes[0]    
                    top_left_x=int(box.xyxy.tolist()[0][0])
                    top_left_y=int(box.xyxy.tolist()[0][1])
                    bottom_right_x=int(box.xyxy.tolist()[0][2])
                    bottom_right_y=int(box.xyxy.tolist()[0][3])
                    extracted_image = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                    new_image=cv.resize(extracted_image,(224,224))
                    encoding =new_image
                    cv.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
                    cv.imshow(name, frame)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                    verified=int(input("ENter 1 to verify else enter 0"))
                    if verified:
                        reg = Register()
                        reg.addPerson(encoding, name)
                        break
            if x == 2:
                super(Start, self).start_face_recognition()
            if x == -1:
                break

st = Start()
st.launch()
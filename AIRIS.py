# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 13:19:16 2021

@author: hp
"""

 # import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import RPi.GPIO as GPIO  
import smbus #import SMBus module of I2C
from time import sleep    
import imutils
import time
import cv2
import math
import speech_recognition as sr
import pyaudio
import os ,time
#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47
TEMP = 0x41

def MPU_Init():
#write to sample rate register
bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)

#Write to power management register
bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)

#Write to Configuration register
bus.write_byte_data(Device_Address, CONFIG, 0)

#Write to Gyro configuration register
bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)

#Write to interrupt enable register
bus.write_byte_data(Device_Address, INT_ENABLE, 1)
       
def read_raw_data(addr):
#Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)
   
        #concatenate higher and lower value
        value = ((high << 8) | low)
       
        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value

def get_direction(Ax,Ay,Az):
        if Ax<1100 and Ax>500:
                print('right')
                robot("right")
        else:
        print('left')
        robot('left')


GPIO.setmode(GPIO.BCM)                     #Set GPIO pin numbering

TRIG = 23                                  #Associate pin 23 to TRIG
ECHO = 24                                  #Associate pin 24 to ECHO

GPIO.setup(TRIG,GPIO.OUT)                  #Set pin as GPIO out
GPIO.setup(ECHO,GPIO.IN)                   #Set pin as GPIO in
def robot(text):
    os.system("espeak ' " + text + " ' " )

r = sr.Recognizer()
def get_speech_command():
    try:
        print('enter anything:')
        robot('tell me sir')
        r.adjust_for_ambient_noise(source,duration=1)
        audio = r.listen(source,2)
        print("System Predicts:"+r.recognize_google(audio))
        text = r.recognize_google(audio)
        print('you said : {}',format(text))
        return text
    except Exception:
        print("Something Wrong")
        robot('try again')
        return "no input given"


           




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

with sr.Microphone(device_index=3) as source:
    while True:
            bus = smbus.SMBus(1) # or bus = smbus.SMBus(0) for older version boards
            Device_Address = 0x68   # MPU6050 device address
            MPU_Init()
            print (" Reading Data of Gyroscope and Accelerometer")

            #Read Accelerometer raw value
            acc_x = read_raw_data(ACCEL_XOUT_H)
            acc_y = read_raw_data(ACCEL_YOUT_H)
            acc_z = read_raw_data(ACCEL_ZOUT_H)
           
            #Read Gyroscope raw value
            gyro_x = read_raw_data(GYRO_XOUT_H)
            gyro_y = read_raw_data(GYRO_YOUT_H)
            gyro_z = read_raw_data(GYRO_ZOUT_H)
           
            #Full scale range +/- 250 degree/C as per sensitivity scale factor
            Ax = acc_x/10
            Ay = acc_y/10 #send me mailok
            Az = acc_z/10
           
            Gx = gyro_x/10
            Gy = gyro_y/10
            Gz = gyro_z/10
            print ("\tAx=%.2f g" %Ax, "\tAy=%.2f g" %Ay, "\tAz=%.2f g" %Az)
            tempRow=(TEMP)
            tempC=(tempRow - 32)*(5/9)
            tempC="%.2f" %tempC
            print("Temp: ")
            print(str(tempC),'degree celsius')
                 
            GPIO.output(TRIG, False)
            print("Waitng For Sensor To Settle")
            time.sleep(2)                        
            GPIO.output(TRIG, True)               
            time.sleep(0.00001)                      
            GPIO.output(TRIG, False)                 
            while GPIO.input(ECHO)==0:               
                pulse_start = time.time()            
            while GPIO.input(ECHO)==1:               
                pulse_end = time.time()              

            pulse_duration = pulse_end - pulse_start 
            distance = pulse_duration * 17150        
            distance = round(distance, 2)            
            if distance > 2 and distance < 400:      
                print("Distance:",distance - 0.5,"cm")
               

                #Print distance with 0.5 cm calibration
            else:
                print("Out Of Range")
                print('byebye')
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]): #continue
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    text=get_speech_command()
                    last_chars=text[-2:]
                    if  "is" in last_chars or "us" in last_chars or "ok" in text or "hello" in text:
                           
                        print(text)
                        print(label)
                        robot(label)
                        get_direction(Ax,Ay,Az)
                        robot(str(distance))
                    if "temperature" in text:
                            robot(str(tempC))
                            robot("degree celsius")
     
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            fps.update()
         
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

       

'''    # Caputure a single frame

         ret, huge_frame = cap.read()
        frame = cv2.resize(huge_frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
# Create the greyscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Add squeres for each face
          for (x, y, w, h) in faces:
                  distancei = (2*3.14 * 180)/(w+h*360)*1000 + 3
                  print (distancei)
#        distance = distancei *2.54
                  distance = math.floor(distancei/2)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                  roi_color = frame[y:y+h, x:x+w]
                 eyes = eye_cascade.detectMultiScale(roi_gray)
              for (ex,ey,ew,eh) in eyes:
                 cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
                 cv2.putText(frame,'Distance = ' + str(distance) + ' Inch', (5,100),font,1,(255,255,255),2)
                 cv2.imshow('face detection', frame)
                    if cv2.waitKey(1) == ord('q'):
                         break
 
 
# Stop the capture
#cap.release()
# Destory the window
#cv2.destroyAllWindows()
'''


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
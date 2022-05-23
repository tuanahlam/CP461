import cv2
import numpy as np
import pytesseract

frameWidth = 640    #Frame Width
franeHeight = 480   # Frame Height

plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500

cap =cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,franeHeight)
cap.set(10,150)
count = 0

while True:
    success , img  = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray,1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"NumberPlate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)
    cv2.imshow("Result",img)
    
    if cv2.waitKey(1) & 0xFF ==ord('s'):  
        cv2.imwrite("Image\IMAGES.jpg",imgRoi)
        #cv2.rectangle(img,(0,200),(640,300),(0,255,0),2)
        # cv2.putText(img,"Scan Saved",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        cv2.imshow("Result",img)
        
        path = 'Image\IMAGES.jpg' 
        # imgPath = cv2.imread(path)
        #Convert to Grayscale
        gray_image = cv2.cvtColor(imgRoi,cv2.COLOR_BGR2GRAY)
        #Remove Noise
        noise_image = cv2.bilateralFilter(gray_image,11,17,17)
        #thresh
        (thresh,thresh_image) = cv2.threshold(noise_image, 120, 255,cv2.THRESH_BINARY)
        cv2.imwrite("Image\IMAGES_thresh.jpg",thresh_image)
        
        
        #OCR
        text = pytesseract.image_to_string(path,lang='tha')
        text = ''.join(e for e in text if e.isalnum())
        print("License Plate : "+text)
        
        path = 'Image\IMAGES_thresh.jpg' 
        #OCR
        textThresh = pytesseract.image_to_string(thresh_image,lang='tha')
        textThresh = ''.join(e for e in textThresh if e.isalnum())
        print("License Plate thresh : "+textThresh)
        
        cv2.waitKey(500)
        
        
        
    

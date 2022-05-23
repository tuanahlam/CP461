import cv2
import pytesseract
import numpy as np


def extract_plate(img):
    image = cv2.imread(img)
    #Convert to Grayscale
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Remove Noise
    noise_image = cv2.bilateralFilter(gray_image,11,17,17)
    #Canny Edge detection
    canny_edge = cv2.Canny(noise_image,100,200)
    #Find contours based on Edges
    contours, new = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    #Initialize license plate contour and x,y coordinates
    contour_with_license_plate = None
    license_plate = None
    x = None
    y = None
    w = None
    h = None

    #Find contour with 4 potential corners and create a Region of Interest around it
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        #This checks id it;s a rectangle
        if len(approx) == 4:
            contour_with_license_plate = approx
            x,y,w,h = cv2.boundingRect(contour)
            license_plate = gray_image[y:y + h , x:x +w]
            break
        
    ## iMAGE PROCESSING 
    imgLicense = gray_image[y:y + h , x:x +w]   
    kernel = np.ones((1, 1), np.uint8)
    dilate_plate = cv2.dilate(imgLicense, kernel, iterations=1)
    erode_plate = cv2.erode(dilate_plate, kernel, iterations=1)
    #thresh
    (thresh,thresh_image) = cv2.threshold(erode_plate, 80, 255, cv2.THRESH_BINARY)
    #Draw License plate
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0),3)

    #crop license plate
    #imageState = cv2.rectangle(image,(x,y+40),(x+w,y+h),(0, 255, 0),3)
    imgLicense_crop = gray_image[y+40:y+h , x:x +w] 
    
    #OCR
    text = pytesseract.image_to_string(license_plate, lang='eng',config='--psm 6')
    text = ''.join(e for e in text if e.isalnum())
    print("License plate : ",text)  
    
    #thresh img OCR
    textThresh = pytesseract.image_to_string(thresh_image, lang='tha',config='--psm 6')
    textThresh = ''.join(e for e in text if e.isalnum())
    print("License plate thresh : ",textThresh) 
    
    #state img OCR
    textState = pytesseract.image_to_string(imgLicense_crop, lang='tha',config='--psm 6')
    textState = ''.join(e for e in textState if e.isalnum())
    print("License plate state : ",textState)
    
  

    cv2.imshow("Original Image",image)
    cv2.imshow("Gray_Image",gray_image)
    cv2.imshow("license_plate",license_plate)
    cv2.imshow("license_plate_Thresh",thresh_image)
    # cv2.imshow("license_plate_Crop",imgLicense_crop)
    # cv2.imshow("Noise_Image",noise_image)
    
    
    cv2.waitKey(0)
    
path = 'LicensePlate/6.jpg'
    
extract_plate(path)



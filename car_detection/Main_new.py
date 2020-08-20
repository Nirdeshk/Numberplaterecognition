
# coding: utf-8

# In[1]:


# Loading Libraries
import cv2
import numpy as np
import time


# In[2]:


classNames = ['car']
COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
inWidth = 640
inHeight = 480
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5


# In[5]:


# VideoCapture
cap = cv2.VideoCapture(1)

col1 = (0,0,0)
col2 = (0,0,0)
col3 = (0,0,0)
t = time.time()
while True:

    _,image = cap.read()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (inWidth,inHeight)), inScaleFactor,
        (inWidth, inHeight), meanVal)
    
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
    net.setInput(blob)
    detections = net.forward()
    cols = image.shape[1]
    rows = image.shape[0]
    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))


    y1 = int((rows - cropSize[1]) / 2)
    y2 = y1 + cropSize[1]
    x1 = int((cols - cropSize[0]) / 2)
    x2 = x1 + cropSize[0]
    image = image[y1:y2, x1:x2]

    cols = image.shape[1]
    rows = image.shape[0]

    
    
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 7:
                obj = 0
                    
                #print(class_id)
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                position = (xLeftBottom, yLeftBottom, xRightTop, yRightTop)
                (cx,cy) = (xLeftBottom+xRightTop)/2 , (yLeftBottom+yRightTop)/2

                #print((xLeftBottom+xRightTop)/2 , (yLeftBottom+yRightTop)/2)
                cv2.circle(image,(int(cx),int(cy)),3,(0,255,0),1)
                if cx < 240:
                    count1 +=1
                
                elif cx > 240 and cx < 430:
                    count2 +=1
                    
                else:
                    count3 +=1
                    
                #cv2.circle(image,(447,63), 5, (255,0,0), -1)

                cv2.rectangle(image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (255, 0, 0))

                label = classNames[obj] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                
    #print(time.time() - t)
    cv2.putText(image, 'time'+str(time.time() - t), (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
    if time.time() - t > 10:
        

        if count1 >= count2 and count1 >= count3:
            col1 = (0,255,0)
            col2 = (0,0,255)
            col3 = (0,0,255)
            t = time.time()
            


        elif count2 >= count1 and count2 >= count3:
            col1 = (0,0,255)
            col2 = (0,255,0)
            col3 = (0,0,255)
            t = time.time()

        else: #count3 >= count1 and count3 >= count2:
            col1 = (0,0,255)
            col2 = (0,0,255)
            col3 = (0,255,0)
            t = time.time()
        
    
    cv2.line(image,(240,150),(430,150),col2,2)
    cv2.line(image,(240,150),(150,300),col1,2)
    cv2.line(image,(430,150),(500,300),col3,2)
    #print(count1,count2,count3)

    #out.write(image)
    cv2.imshow('image',image)
    #cv2.imshow('ithre',thresh)
    if cv2.waitKey(41) == 27:
        break

cap.release()
cv2.destroyAllWindows()




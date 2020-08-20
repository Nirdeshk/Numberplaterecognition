import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import PossiblePlate
# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

img  = cv2.imread("1.png")
listOfPossiblePlates = DetectPlates.detectPlatesInScene(img)
print(img)




cv2.imshow('windows', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

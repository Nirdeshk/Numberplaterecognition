import cv2
im = imread("/Users/karna/Desktop/Licience_plate_reco/newim.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
axis("off")
title("newim")
imshow(im_gray, cmap = 'gray')
show()

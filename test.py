#!/usr/bin/python
import cv2
import numpy as np
import sys

if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except:
        fn = '../data/road/test_Adachi_00001.jpg'

    """
    img = cv2.imread(fn)
    #flag, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original',img)

    gray_filtered = cv2.inRange(gray, 190, 255)
    cv2.imshow('Gray filtered',gray_filtered)

    def update(_):
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)

        vis = img.copy()
        vis = np.uint8(vis/2.)
        vis[edge != 0] = (0, 255, 0)
        cv2.imshow('edge', vis)

    cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', 3240, 5000, update)
    cv2.createTrackbar('thrs2', 'edge', 5000, 5000, update)

    update(None)
    """

    img = cv2.imread(fn, 0)

    """
    adaptive threshold
    """
    img = cv2.medianBlur(img,5)
    edge = cv2.Canny(img, 3140, 5000, apertureSize=5)

    
    vis = img.copy()
    vis = np.uint8(vis/2.)
    vis[edge != 0] = 255
    #img = vis
    

    thrs = 50
    ret,th1 = cv2.threshold(vis,thrs,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    th3 = np.uint8(th3/5.)
    th3[edge != 0] = 0

    titles = ['Original Image', 'Global Thresholding (v = '+str(thrs)+')',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    
    for i in xrange(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


    cv2.waitKey(0)
    cv2.destroyAllWindows()

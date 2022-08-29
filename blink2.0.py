import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from pynput.keyboard import Key, Controller

keyboard = Controller() 

cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)

leftEyeLandmarks = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

plotY = LivePlot(640, 360, [0,40], invert=True)

while True:
    # Capturing frame
    retval, frame = cap.read()

    # Exit the application if frame not found
    if not retval:
        break

    # Getting face recognition mesh     
    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        # Draws every landmark on the left eye
        for i in leftEyeLandmarks:
            cv2.circle(frame, face[i], 5, (255,0,255), cv2.FILLED)

        # Vertical and Horizontal keypoints
        topEyelid = face[159]
        bottomEyelid = face[23]
        leftCorner = face[130]
        rightCorner = face[243]
        
        # Vertical distance
        lengthVer,_ = detector.findDistance(topEyelid, bottomEyelid)

        # Horizontal distance
        lengthHor,_ = detector.findDistance(leftCorner, rightCorner)

        # Draws the vertical line
        cv2.line(frame, topEyelid, bottomEyelid, (0, 200, 0), 3)

        # Draws the horizontal line
        cv2.line(frame, leftCorner, rightCorner, (0, 200, 0), 3)

        # Blinking ratio
        ratio = int((lengthVer/lengthHor) * 100)

        # If the eyelids are closed, press the spacebar
        if ratio < 27:  
            keyboard.press(Key.space)
        # If the eyelids are open, release the spacebar                   
        if ratio > 27:                                                                                 
            keyboard.release(Key.space)          

        # Shows the frame and the plot on top of each other
        imgPlot = plotY.update(ratio)            
        frame = cv2.resize(frame, (640, 480))
        frameStack = cvzone.stackImages([frame,imgPlot], 1, 1)

    else:
        # Only draw the frames
        frame = cv2.resize(frame, (640, 480))
        frameStack = cvzone.stackImages([frame, frame], 1, 1)

    # Opens video captured window
    cv2.imshow("BlinkDetection2.0", frameStack)

    # WaitKey(1) will wait for a keyPress for just 1 milliseconds and it will
    # continue to refresh and read frame from your webcam using cap.read().
    key = cv2.waitKey(1)

    # If the escape key is pressed, close the application 
    if key == 27:
        break 

cap.release()
cv2.destroyAllWindows()
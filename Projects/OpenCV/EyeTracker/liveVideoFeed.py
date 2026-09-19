import cv2

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        print("Error Chopped Face Not Found")
        break

    #Invert horizontally (flipCode 1 = horizontal mirror)
    frame = cv2.flip(frame,1)

    frame = cv2.convertScaleAbs(frame, alpha= 2, beta= 10)
    
    #show live feed 
    cv2.imshow("Web Cam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
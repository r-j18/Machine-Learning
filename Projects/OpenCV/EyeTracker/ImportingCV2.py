import cv2

cap = cv2.VideoCapture(0) #0 is device 0 (default webcam)

#ret = true if succesful, frame = image turned into numpy array
ret, frame = cap.read()

print("Succesfull: ", ret)
print("Frame Shape (height, width, channel(BGR)): ", frame.shape)

#releases camera for other apps to use
cap.release()

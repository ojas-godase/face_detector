import cv2

# Load some pre-trained data on face frontals from opencv(haar sascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam
webcam = cv2.VideoCapture(0)



# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read , frame = webcam.read()

    # Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw a rectangle around the face
    # Using the for loop to show rectangle around multiple faces
    for faces in face_coordinates:
        (x,y,w,h) = faces
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2)
    # Show the frame
    cv2.imshow('Photo' , frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
webcam.release()
print("Code working")



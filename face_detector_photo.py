import cv2

# Load some pre-trained data on face frontals from opencv(haar sascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('multiplefaces.jpg')

# Convert to grayscale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw a rectangle around the face
for faces in face_coordinates:
    (x,y,w,h) = faces
    cv2.rectangle(img, (x,y) , (x+w,y+h) , (0,255,0) , 2)


cv2.imshow('Photo' , img)
cv2.waitKey(20000)
print("Code working")
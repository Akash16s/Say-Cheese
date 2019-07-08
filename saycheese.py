#importing the library
import cv2
import dlib
from scipy.spatial import distance 
from imutils import face_utils


def smiledetector(coordinates):
	val1 = distance.euclidean(coordinates[3],coordinates[9])
	val2 = distance.euclidean(coordinates[2],coordinates[10])
	val3 = distance.euclidean(coordinates[4],coordinates[8])
	average = (val1+val2+val3)/3
	length = distance.euclidean(coordinates[0],coordinates[6])
	return average/length

cap = cv2.VideoCapture(0) #Creating the object

detect = dlib.get_frontal_face_detector() # Object Created
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
	ret, img = cap.read()  # Reading images

	if not ret:           #Checking that image is recieved
		break

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	boxes = detect(gray)  # Detection the location of faces
	#print(boxes)
	for i,d in enumerate(boxes):
		shape = predictor(gray, d) #Applying predictor on each face we detect
		shape = face_utils.shape_to_np(shape) # Converting the object to numpy array
		mar = smiledetector(shape[48:68]) # passing coordinates
		#print(mar)
		if(mar > 0.30 and mar< 0.39):   #determining smiling or neutral
			print("neutral")
		elif(mar > 0.39):
			print("smiling")
			cv2.imwrite("image.png",img)


	cv2.imshow("Image", img)    # Showing the video feed
	key = cv2.waitKey(1) & 0xFF  
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
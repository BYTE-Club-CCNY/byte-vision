import cv2
import mediapipe as mp #these are the dependencies we need to install. be sure to check #README.md for the installation instructions

mp_drawing = mp.solutions.drawing_utils #drawing utilities for drawing landmarks on the image
mp_pose = mp.solutions.pose
pose = mp_pose.Pose() #init pose 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2) #init hands, and limit to 2 hands

cap = cv2.VideoCapture(0) #this opens the default camera. you can change the index to open a different camera in case u have multiple cameras connected

while cap.isOpened(): #this loop will run until the camera is closed or the user presses 'q'
    ret, frame = cap.read() 
    if not ret: #if the frame is not read correctly, we break the loop
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert the frame to RGB
    pose_results = pose.process(frame_rgb)
    if pose_results.pose_landmarks: #if pose landmarks are detected
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS) #draw only the pose landmarks on the image
    

        #now, if hands are detected, we draw them too
        hand_results = hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    #try out these different filters on the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray scale is useful for blocking out colors
    
    '''
    these two filters are useful for smoothing the image and removing noise for classification models
    and by classification, i mean models that classify the image as a whole, like object detection models or image classification models
    '''
    gaussian = cv2.GaussianBlur(frame, (15, 15), 0)  
    filtered = cv2.bilateralFilter(frame, 9, 75, 75)  
    
    edges = cv2.Canny(frame, 100, 200) #this is useful for edge detection and can be used for pose estimation models. removes a lot of noise and helps the model focus on objects in the image

    cv2.imshow("Pose+Hands Tracking - Press Q to Stop", frame)
    #uncomment the following lines to see the different filters in action
    #cv2.imshow("Gray", gray)
    #cv2.imshow("Gaussian", gaussian)
    #cv2.imshow("Filtered", filtered)
    #cv2.imshow("Edges", edges)
    if cv2.waitKey(1) & 0xFF == ord('q'): #our stopping condition
        break

cap.release()
cv2.destroyAllWindows()

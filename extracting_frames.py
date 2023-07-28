import os # os module, which provides a way to interact with the operating system, allowing operations like file and directory manipulation.
import cv2
from math import floor #used later in the code for calculating the floor value of the rate of frames.

def extract_frames(frames_path, video_file): # defines a function called extract_frames that takes two parameters: frames_path (the path to save the extracted frames) and video_file (the path of the video file from which frames need to be extracted).
    if len(os.listdir(frames_path)) == 0: #checks if the frames_path directory is empty. If it is empty, the code proceeds to extract frames from the video; otherwise, it indicates that the directory is not empty.
        count = 0 # initializes a variable count to keep track of the frame number.
        video_object = cv2.VideoCapture(video_file)  # creates a VideoCapture object to read the input video file.
        rate_of_frame = video_object.get(5) #Gets the frame rate of the video using the get method with the argument 5.the argument 5 corresponds to the property code CV_CAP_PROP_FPS, which represents the frames per second (FPS) of the video. 
        while(video_object.isOpened()): #initiates a loop to read each frame from the video until the video is fully read or there are no more frames left.
            frame_id = video_object.get(1)#The argument 1 corresponds to the property code CV_CAP_PROP_POS_FRAMES, which represents the index of the next frame in the video stream to be captured.
            content, frame = video_object.read() #eads the current frame of the video and stores it in the variable frame, and content indicates whether the frame was successfully read or not.
            if (content != True): # frame was not successfully read, the loop breaks, indicating the end of the video.
                break
            #checks if the current frame's index is a multiple of the floor value of the frame rate. If it is, the frame will be extracted; otherwise, it will be skipped to downsample the number of frames extracted.
            if (frame_id % floor(rate_of_frame) == 0):
                filename ="frame%d.jpg" % count #generates a filename for the extracted frame using the current value of count.
                count+=1
                cv2.imwrite(frames_path+filename, frame) #rites the extracted frame to the specified path with the generated filename.
        video_object.release() #releases the video object after processing.
        print ("Done!", count)

    else:    
        print("Directory is not empty")
    
# defines a function that takes a path to an image as input and returns the image and detected faces within the image using the Haar cascade classifier for face detection.
def face_extract_using_CV(path):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #creates a Haar cascade classifier for face detection using the pre-trained frontal face XML file from OpenCV's data.
    image = cv2.imread(path)
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converts the color space of the image from BGR to RGB using OpenCV's cvtColor function.
    faces = faceCascade.detectMultiScale(
    color,
    scaleFactor=1.2,
    minNeighbors=10,
    minSize=(64,64),
    flags=cv2.CASCADE_SCALE_IMAGE
    ) #detects faces within the color image using the detectMultiScale method of the Haar cascade classifier with specified parameters for scaleFactor, minNeighbors, and minSize.
    
    return image, faces    #returns the original image (image) and the bounding boxes of detected faces (faces).
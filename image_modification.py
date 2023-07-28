import cv2
from numpy import asarray #asarray function from NumPy, which is used to convert lists or tuples to arrays.
from skimage.transform import resize #resize function from scikit-image, which is used to resize images.
from os import listdir
from os.path import isdir

#This function is used to extracted face from an image and the output is the raw image along with the face/faces.

def face_extract_using_CV(path): #Defines a function that takes the path of an image as input.

    image = cv2.imread(path)
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #initializes the Haar Cascade classifier for face detection.
    faces = faceCascade.detectMultiScale( #Detects faces in the image using the detectMultiScale method of the Haar Cascade classifier.
    color,
    scaleFactor=1.2, #Parameter specifying how much the image size is reduced at each image scale. It helps in detecting faces at multiple scales.
    minNeighbors=10,# Parameter specifying how many neighbors each rectangle should have to be considered as a face
    minSize=(64,64), #Minimum possible size for a detected face.
    flags=cv2.CASCADE_SCALE_IMAGE #flag that indicates how the scale factor is applied. In this case, it is set to CASCADE_SCALE_IMAGE, which means the scale factor is applied to the image size, not to the detection window size.
    )

    return image, faces
    #Returns the original image and the coordinates of the detected faces as a tuple. Each face is represented by a set of (x, y, w, h)
    #  coordinates, where (x, y) is the top-left corner of the face rectangle, and (w, h) are the width and height of the rectangle, respectively.


#This function is used to take face image and outputs array of pixels

def resize_face(filename, size):
    
    image = cv2.imread(filename)
    resize_image = resize(image, size)
    pixels = asarray(resize_image)

    return pixels

#This function is used to take person's faces from the respective folders and pass on to face modeling function

def faces_from_folders(folder, size):
    
    extracted_faces = []
    
    for filename in listdir(folder):
        image_path = folder + filename
        face = resize_face(image_path, size)
        extracted_faces.append(face)
        
    return extracted_faces


#This function is used to iterate over each person's face from the respective folders and output an array of person's face pixels and the corresponding label

def loading_images(folder, size):
    X = [] ; y = []
    
    for sub_folder in listdir(folder):
        
        image_path = folder + sub_folder + '/'
        if not isdir(image_path):
            continue
        faces = faces_from_folders(image_path, size)
        labels = [sub_folder for _ in range(len(faces))]
        
        print('Loaded %d samples for character: %s' % (len(faces), sub_folder))
        
        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)

from keras.models import load_model #load_model function from Keras, which is used to load a pre-trained deep learning model.
#imporing necessary libraries
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer

#importing necessary functions from ML_pipepline
from ML_pipeline.download_video import download_video
from ML_pipeline.create_new_folder import create_new_folder
from ML_pipeline.extracting_frames import extract_frames
from ML_pipeline.image_modification import loading_images
from ML_pipeline import embedding_encoding
from ML_pipeline.facenet_prediction import accuracy
from ML_pipeline.facenet_prediction import predict
from ML_pipeline.facenet_prediction import facenet_image_prediction
from ML_pipeline.facenet_prediction import facenet_video_prediction 

#run the following snippet only if video download and extraction of frames has to be done.
 #https://www.youtube.com/watch?v=NzOTuh63eVs
# Download video from you tube
'''
download_video(video_link = "https://www.youtube.com/watch?v=miLBEMYAaH8", path_to_store='../input/video')

# Create new folder to save frames 
create_new_folder(path = '../input/frames_path')

# Extract frames from video and saving them in frames_path folder
extract_frames(frames_path= '../input/frames_path/', video_file="../input/video/Friends - Monica and Chandlers Wedding.mp4")

'''
# load train dataset
#Loads training images and their corresponding labels from the specified folder (../input/train/) using the loading_images function. The images are resized to (160, 160) dimensions.
X_train, y_train = loading_images("input/train/", (160,160))
X_test, y_test = loading_images("input/val/", (160,160))

#loading facenet model
model = load_model('prebuilt_models/facenet_keras.h5') #Loads the pre-trained FaceNet model from the specified file

# make training dataset
X_train_embedded = embedding_encoding.embedded_array(model, X_train) #Generates FaceNet embeddings for the training images (X_train) using the embedded_array function from the embedding_encoding module.
X_test_embedded = embedding_encoding.embedded_array(model, X_test)
# Normalizer
l2_encoder = Normalizer(norm='l2') #Initializes a Normalizer with L2 normalization, which is used to normalize the embedding vectors.

# normalized embedding vectors
#Normalizes the FaceNet embedding vectors for both training and validation sets using L2 normalization.
vectorized_data  = embedding_encoding.vectorize_vectors(l2_encoder, [X_train_embedded, X_test_embedded])
#xtracts the normalized training data from the vectorized_data.
normalized_train = vectorized_data[0]
normalized_test = vectorized_data[1]


# lable encoded train labels
#Encodes the training and validation labels using LabelEncoder from the embedding_encoding module. It returns the label encoder and the encoded training and validation labels.
label_encoder, encoded_train, encoded_test = embedding_encoding.encode_target(y_train, y_test)

# mapping the dictionary for the person name with the lable encoded
label_map = {} #Initializes an empty dictionary to map encoded labels to person names.
#iterates over each person's name in the label encoder's classes.
for index,name in enumerate(label_encoder.classes_):
    label_map[index] = name #Maps the index (encoded label) to the person's name in the label_map dictionary.

# fit model
#Initializes a Support Vector Classifier (SVC) with a linear kernel and probability estimates enabled.
ml_model = SVC(kernel='linear', probability=True)
ml_model.fit(normalized_train, encoded_train) #Trains the SVC model using the normalized training data and encoded training labels.

#prediction on validation data
predicted_val=predict(normalized_test, ml_model) #Makes predictions on the validation data using the trained SVC model and the normalized validation data.
# Computes the accuracy score of the predictions on the validation data compared to the encoded validation labels.
accuracy_score= accuracy(encoded_test, predicted_val)
print('Accuracy score on validation data is ',accuracy_score)

#The subsequent code performs FaceNet frame prediction and video prediction using the trained SVC model and the pre-trained FaceNet model.
#  The frames are processed, and predictions are made for each frame. The results are stored in the output folder.
# Facenet frame prediction
folder= 'input/test_frames/'
size=(160,160)
output='../output/'
facenet_image_prediction(folder,output, l2_encoder, ml_model, model, label_map, size)


#Facenet video prediction
print('press Q on keyboard to exit from video window')
video_file_path='input/video//Friends - Monica and Chandlers Wedding.mp4'
facenet_video_prediction(video_file_path, l2_encoder,label_map,model,ml_model,size)


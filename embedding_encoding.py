#The given code defines four functions used for generating embeddings from a pre-trained FaceNet model and preparing the data for training a machine learning model.
#  Here's a brief explanation of each function:
import numpy as np
from numpy import asarray, expand_dims
from sklearn.preprocessing import LabelEncoder


#This function is used to predict 1D 128 vector embeddings from a pre-trained facenet model
# function generates a 128-dimensional embedding vector from a pre-trained FaceNet model for a given input face image represented by pixels.
#  It normalizes the input pixels, expands the dimensions to match the model's input shape, and then uses the model to predict the embedding.
def embedding_generation_from_facenet(model, pixels):

    pixels            = pixels.astype('float32')
    pixels_mean       = np.mean(pixels)
    pixels_deviation  = np.std(pixels)
    normalized_pixels = (pixels - pixels_mean) / pixels_deviation
    transformed_pixel = expand_dims(normalized_pixels, axis=0)
    embeddings = model.predict(transformed_pixel)
    
    return embeddings[0]



#This function is used to get embeddings from each face pixels from a training set
#function takes an array of face images represented by array_data and generates embeddings for each face image using the embedding_generation_from_facenet function.
#  It returns an array containing the embeddings for all the input face images.
def embedded_array(model, array_data):
    
    embedding_list = []
    for face_pixels in array_data:
        embedding = embedding_generation_from_facenet(model, face_pixels)
        embedding_list.append(embedding)
    embedding_list = asarray(embedding_list)
    return embedding_list



#This function is used to normalize the embedding vectors
# function normalizes the embedding vectors using the provided encoder. It is commonly used for scaling the embeddings to have zero mean and unit variance.
def vectorize_vectors(encoder, dataX):
    
    normalized_data = []
    for data_x in dataX:
        normalized_data.append(encoder.transform(data_x))
    
    return normalized_data


#This function is used to label encode the train and test labels
# function performs label encoding on the training and test labels (datay1 and datay2). It uses the LabelEncoder from sklearn.preprocessing
#  to convert the labels into numerical values, which can be used for training a machine learning model.
def encode_target(datay1, datay2):

    label_encoder = LabelEncoder()
    label_encoder.fit(datay1)
    datay1_train  = label_encoder.transform(datay1)
    datay2_test   = label_encoder.transform(datay2)
    
    return label_encoder, datay1_train, datay2_test
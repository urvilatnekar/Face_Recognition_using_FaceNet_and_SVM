import os
import errno

#function to create new folder given a path 
def create_new_folder(path):
    try:
        #making a new folder
       os.makedirs(path) #used to create a new folder/directory at the given path.
    except OSError as e: #If the path already exists, it will raise an OSError with errno.EEXIST indicating that the folder already exists.
        if e.errno != errno.EEXIST: #This line checks if the error code associated with the caught exception is not errno.EEXIST, which stands for "File exists."
             #If the error code is not errno.EEXIST, it means that the folder could not be created due to some other error.
            raise 
#the raise statement is executed, which re-raises the caught exception, propagating it up the call stack to be handled by higher-level exception handlers or
#  terminating the program if there are no higher-level exception handlers.
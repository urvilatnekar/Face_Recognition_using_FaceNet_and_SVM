from pytube import YouTube

def download_video(video_link, path_to_store):
    try: 
        # object creation using YouTube
        yt = YouTube(video_link) #creates a YouTube object by passing the video_link, which is the URL of the YouTube video that needs to be downloaded.
    except: 
        # to handle exception due to connection error
        print("Connection Error") #f there is an exception during the creation of the YouTube object, it will catch any connection error that may occur and print "Connection Error."
   
    stream = yt.streams.get_by_itag('22') #retrieves the stream (video format) with the specified itag value '22'. The itag value '22' represents the video format with 720p resolution.
    
    #taking highest resolution
    stream = yt.streams.get_highest_resolution() #retrieves the stream with the highest resolution available for the video.
    
    try: 
        # downloading the video 
        stream.download(path_to_store) #downloads the video stream to the specified path_to_store, which is the folder path where the downloaded video will be saved.
        print(yt.title,'is downloaded and stored in ', path_to_store) # yt.title refers to the title of the YouTube video.
    except: 
        print("Error while downloading the video!") 
        #If there is any error during the video download process, it will catch the exception and print "Error while downloading the video!"


    
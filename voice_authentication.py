
# coding: utf-8

# # Importing libraries

# In[1]:


from scipy.io.wavfile import read
from sklearn.mixture import GMM 
from featureextraction import extract_features
import speech_recognition as sr
import pickle
import shutil
import os 
import numpy as np
import time
import pyaudio
import wave
import warnings
warnings.filterwarnings("ignore")


# # Making data runtime

# In[2]:


def making_audio_files():
    global source1
    for i in range(1,8):
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 2
        WAVE_OUTPUT_FILENAME = "file"+str(i)+".wav"

        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
        print("say again")
        print("recording...")
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("finished recording")

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        #Saving the file
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
        time.sleep(1.0)


# In[3]:


#making_audio_files()
def shifting_of_files():
    for i in range(1,8):
        #Moving file from one location to other location
        shutil.move('/home/keshav/Desktop/voice_recognisation/file'+str(i)+'.wav', '/home/keshav/Desktop/voice_recognisation/trainingData/user-1/file'+str(i)+'.wav')
#training()


# # Training Model
# 

# In[4]:


def training_model():
    #path to training data
    source = "trainingData/"   

    #path where training speakers will be saved
    dest = "Speakers_models/"
    train_file = "trainingDataPath.txt"        
    file_paths = open(train_file,'r')
    print(type(file_paths))

    count = 1
    # Extracti(ng features for each speaker (5 files per speakers)
    features = np.asarray(())

    print(file_paths)
    for path in file_paths:
        path = path.strip()   
        print(path)

        # read the audio
        sr,audio = read(source + path)

        # extract 40 dimensional MFCC & delta MFCC features
        vector   = extract_features(audio,sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 7:    

            gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)        

            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("-")[0]+".gmm"
            print(picklefile)

            pickle.dump(gmm,open(dest + picklefile,'wb'))
            print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)    
            features = np.asarray(())
            count = 0
            break
        count = count + 1


# # Finding max and min value of log_likelihoods

# In[5]:


def getting_log_likelihood():
    global maximum, minimum
    #path to training data
    source   = "trainingData/"

    #path where training speakers will be saved
    modelpath = "Speakers_models/"

    gmm_files = [os.path.join(modelpath,fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]

    #Load the Gaussian gender Models
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
                 in gmm_files]

    error = 0
    total_sample = 0.0
    train_file = "trainingDataPath.txt"
    file_paths = open(train_file,'r')
    f = 0
    # Read the test directory and get the list of test audio files
    list1 =[]
    for path in file_paths:
        total_sample += 1.0
        path = path.strip()
        print(path)
    #print("Testing Audio : ", path)
        sr,audio = read(source + path)
        vector   = extract_features(audio,sr)

        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
            f += 1
            list1.append(scores.sum())

        print(log_likelihood[i])
        if (f == 7):
            break

    print(len(list1))
    maximum = max(list1)
    minimum = min(list1)
    return maximum, minimum


# # Testing Runtime

# In[6]:


def testing_runtime():
    
    global source1

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "file.wav"
 
    audio = pyaudio.PyAudio()
 
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print("say again")
    print("recording...")
    frames = []
 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    #Saving the file
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    #Moving file from one location to other location
    shutil.move('/home/keshav/Desktop/voice_recognisation/file.wav', '/home/keshav/Desktop/voice_recognisation/SampleData/file.wav')
    
    #time for prediction
    predict_file()
    


# In[7]:


def predict_file():
    global source1, maximum, minimum

    sr,audio = read(source1)
    vector   = extract_features(audio,sr)
    #path to training data
    source   = "SampleData/"   

    #path where training speakers will be saved
    modelpath = "Speakers_models/"

    gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]
    #Load the Gaussian gender Models
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]
    
    log_likelihood = np.zeros(len(models)) 
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    print(log_likelihood[i])
    if(minimum < log_likelihood[i] < maximum):
        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
        check_command(speakers,winner)
    else:
        print('not authorise please try again')
        testing_runtime()
        time.sleep(0.5)
        #winner = None
    time.sleep(1.0)


def check_command(speakers,winner):
    import speech_recognition as sr
    global source1
    
    #Check for authentication
    if speakers[winner] == 'user':
        AUDIO_FILE = source1    
        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.listen(source)
        try:
            command = r.recognize_google(audio).lower()
            print('You said: ' + command + '\n')
        #loop back to continue to listen for commands if unrecognizable speech is received
        except sr.UnknownValueError:
            print('....')

    if  'unlock the system' in command:
        os.system('cheese')
    else:
        print('keyword does not match')


# In[ ]:


global source1, maximum, minimum
source1 = '/home/keshav/Desktop/voice_recognisation/SampleData/file.wav'

if __name__ == "__main__":
    print('''
            press 0 to retrain the model
            press 1 to unlock the system by your trained voice
          '''
         )
    choice = int(input())
    if choice == 0:
        making_audio_files()
        shifting_of_files()
        print("Your Model is Training ........")
        training_model()
        maximum, minimum = getting_log_likelihood()
        print("Training is Completed........")
    elif choice == 1:
        maximum, minimum = getting_log_likelihood()
        testing_runtime()
    else:
        print("wrong choice is entered")


from passlib.hash import argon2
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from getpass import getpass
from cryptography.fernet import Fernet
import cv2
import face_recognition as fr
import os
from time import sleep
import pyaudio
import wave
import os
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn import mixture
from scipy.io.wavfile import read
from python_speech_features import mfcc


def extractfeatures(audio,rate):    
    mfcc_feat = mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True, nfft=1103)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculatedelta(mfcc_feat)
    #combining both mfcc features and delta
    combined = np.hstack((mfcc_feat,delta)) 
    return combined

def calculatedelta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def voiceadd(mastername):
    #Voice authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    print("Starting Voice Biometrics...")
    sleep(2)
    source = "./voice_database/" + mastername + '/'
    os.mkdir(source)
    for i in range(3):
        audio = pyaudio.PyAudio()
        j = 3
        while j>=0:
            sleep(1)
            os.system('clear')
            print("Speak your name in {} seconds".format(j))
            j-=1
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
        print("Recording...")
        sleep(2)
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        # saving wav file of speaker
        waveFile = wave.open(source + '/' + str((i+1)) + '.wav', 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        print("Done")
        dest =  "./gauss_models/"
        count = 1
        for path in os.listdir(source):
            path = os.path.join(source, path)
            features = np.array([])
            # reading audio files of speaker
            (sr, audio) = read(path)
            # extract 40 dimensional MFCC & delta MFCC features
            vector   = extractfeatures(audio,sr)
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
            # when features of 3 files of speaker are concatenated, then do model training
            if count == 3:    
                gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
                gmm.fit(features,y=None)
                # saving the trained gaussian model
                pickle.dump(gmm, open(dest + mastername + '.GaussianMixture', 'wb'))
                print(mastername + "'s voice added successfully") 
                features = np.asarray(())
                count = 0
            count = count + 1

def voicerecog(mastername):
    # Voice Authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 4
    FILENAME = "./test.wav"
    print("Starting Voice Authentication...")
    sleep(2)
    audio = pyaudio.PyAudio()
    j = 3
    while j>=0:
        sleep(1)
        os.system('cls' if os.name == 'nt' else 'clear')#Remove
        print("Please Speak in {} seconds".format(j))
        j -= 1
    # start Recording
    print("Recording . . .")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)   
    sleep(2)
    frames = []
    print("Finished Recording")
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # saving wav file 
    waveFile = wave.open(FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    modelpath = "./gauss_models/"
    gmm_files = [os.path.join(modelpath,fname) for fname in 
                os.listdir(modelpath) if fname.endswith('.GaussianMixture')]
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".GaussianMixture")[0] for fname 
                in gmm_files]
    if len(models) == 0:
        print('There Are No Users In the Database')
        return
    #read test file
    sr,audio = read(FILENAME)
    # extract mfcc features
    vector =extractfeatures(audio,sr)
    log_likelihood = np.zeros(len(models)) 
    #checking with each model one by one
    for i in range(len(models)):
        gmm = models[i]         
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    pred = np.argmax(log_likelihood)
    identity = speakers[pred]
    # if voice not recognized than terminate the process
    if identity != mastername:
            print("Not Recognized! Try again...")
            return False
    else:
        print("User Authenticated...")
        sleep(2)

def facerecog(mastername):
    print('Starting Facial Recognition...')
    sleep(2)
    print('Analyzing Face...')
    x = 0
    # Loads image for existing user to compare
    base_image = fr.load_image_file('faces/' + mastername + '.png')
    encode_face = fr.face_encodings(base_image)[0]
    while True:
        ramp_frames = 30
        video_capture = cv2.VideoCapture(0)
        #allows cam to adjust brightness
        for i in range(ramp_frames):
            temp = video_capt2ure.read()
        ret ,frame = video_capture.read()
        # Writes resulting frame to compare     
        cv2.imwrite('loaded_image/image.png', frame)
        video_capture.release()
        sample_image = fr.load_image_file('loaded_image/image.png')
        x+=1
        #Times Out if face not detected 
        if x == 10:
            print('No Face Detected...')
            print('Terminating...')
            exit(-1)
        #Checking for index error due to face not correctly in camera
        try:
            encode_sample_face = fr.face_encodings(sample_image)[0]
            break
        except IndexError as e:
            continue
     #Compares face with 0.6 tolerance for match
    result = fr.compare_faces([encode_face], encode_sample_face)
    resultstring = str(result)
    os.remove('loaded_image/image.png')
    if resultstring == "[True]":
        print("User Authenticated...")
    else:
        print("Face Not Recognised...")
        return 1

def newpicture(mastername):
    faceCascade = cv2.CascadeClassifier("/home/zacky/Desktop/PasswordProject/haarscascade_frontal_face_defaults.xml")
    print('Please look into the camera for face recognition')
    print('Analyzing Face...')
    while True:
        cam = cv2.VideoCapture(0)
        while(1):
            # Capture frame-by-frame
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #needed for cascadeclassifier
            faces = faceCascade.detectMultiScale(gray, 1.1, 3)
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                # Display the resulting frame
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(20) == ord('x'):
                # Saves face for user
                cv2.imwrite('faces/' + mastername + '.png', frame)
                cv2.destroyAllWindows()
                break
        cam.release()
        try:
            base_image = fr.load_image_file('faces/' + mastername + '.png')
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
            face_img = fr.face_locations(base_image)[0]
            break
        except IndexError as e:
            print('Face Not Found')
            print('Retrying...')
            continue
    print('Face Saved...')


def generatekey(mastername, masterpass):
    masterpass = masterpass.encode() #convert to bytes
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
        )
    key = base64.urlsafe_b64encode(kdf.derive(masterpass))
    with open("./" + mastername + "/key.txt",'wb') as fh:
        fh.write(key)
    print("key derived from MasterPassword:")
    print(key)

def encryptmessage(mastername, string):
    with open("./" + mastername+"/key.txt", 'rb') as fh:
        key = fh.read()
    f = Fernet(key)
    encrypted = f.encrypt(string.encode())
    print("The encrypted string:")
    print(encrypted)
    print("adding the following information:")
    print(string)
    with open("./"+ mastername + "/passwords.txt",'a+') as fh:
        fh.write(encrypted.decode() + "\n")
    print("Log in information added")
    return

def decryptmessage(mastername):
    passlist = []
    with open("./" + mastername+"/key.txt", 'rb') as fh:
        key = fh.read()
    f = Fernet(key)
    with open("./"+ mastername + "/passwords.txt",'r') as fh:
        for line in fh:
            encrypted = line
            passlist.append((f.decrypt(encrypted.encode()).decode()))
        print(passlist)

def usermenu(mastername):
    while True:    
        print("Menu options:")
        print("\n\t\t1. List accounts")
        print("\n\t\t2. Add account")
        print("\n\t\t3. Logout")
        try:
            option = int(input("\nPlease enter the number of the option you would like: "))
        except ValueError as e:
            print('Unrecognised Command')
            print('Please Enter Recognised Command')
            continue
        if option == 1:
            decryptmessage(mastername)
            continue
        if option == 2:
            website = input("please enter the name of the website or service: ")
            accountname = input("please enter the account name or mastername: ")
            password = getpass("please enter the password: ")
            string = (website +":"+ accountname +":"+password)
            encryptmessage(mastername, string)
            continue
        if option == 3:
            print('Logging Out...')
            return
        else:
            print('Unrecognised Command')
            print('Please Enter Recognised Command')
            continue
def startmenu():
    print("Menu options:")
    print("\n\t\t1. Create new User")
    print("\n\t\t2. Log in as existing user")
    print("\n\t\t3. Exit Program")
    try:
        option = int(input("\nPlease enter the number of the option you would like: "))
    except ValueError as e:
        print('Unrecognised Command')
        print('Please Enter Recognised Command')
        return
    if option == 1:
        print("\nCreating new User:")
        mastername = input("\n\t\tPlease enter a Master Name for your account: ")
        masterpass = getpass("\n\t\tPlease enter the Master Password for your password list: ")
        hash = argon2.hash(masterpass)
        masterpath = "./" + mastername
        if not os.path.exists(masterpath):
            os.makedirs(masterpath)
        with open(masterpath + "/hash.txt",'w') as fh:
            fh.write(hash)
        generatekey(mastername, masterpass)
        newpicture(mastername)
        voiceadd(mastername)
        return
    if option == 2:
        mastername = input("\nPlease enter your Master Name: ")
        masterpass = getpass("Please enter the Master Password: ")
        try:
            with open("./" + mastername + "/hash.txt", 'r') as fh:
                hash = fh.read()
            result = argon2.verify(masterpass, hash)
        except FileNotFoundError as e:
            print('Incorrect mastername or Password')
            return
        if result == True:
            x = facerecog(mastername)
            if x == False:
                    return
            else:   
                x = voicerecog(mastername)
                if x == False:
                    return
                else:
                    usermenu(mastername)
        else:
            print('Incorrect mastername or Password')
    if option == 3:
        print('Terminating...')
        exit(-1)
    else:
        return
while True:
    startmenu()

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from .models import User
from .models import RationDetails
from django.contrib.auth.hashers import make_password, check_password
import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from keras.models import load_model
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
import cv2
import mediapipe as mp 
from PIL import Image
import glob
import random
from matplotlib import pyplot as plt
import matplotlib.figure
from datetime import date
from playsound import playsound

today = date.today()

# Create your views here.
context = {
        'login_req': False,
    }
def home(request):
    
    template = loader.get_template('home.html')
    ration_details = RationDetails.objects.filter(ration_card='13').all()
    ration = {
        'ration_details': ration_details,
   	}
    for x in ration['ration_details']:
        print(x.received_date)
    #print(ration)
    if context['login_req'] == False and request.method == "GET":
       print("Admin")
       print("Admin test")
       filename = record_audio_test()
       identity = test_model(filename)
       print(identity)
       if identity == 'Admin':
           context.update({'login_req': True})
           playsound('welcome.mp3')
           return HttpResponse(template.render(context, request))

           
    return HttpResponse(template.render(context, request))

    
def password(request):
    
    template = loader.get_template('home.html')
    
    if request.method == 'GET':        
        ration_card = request.session['ration_card']
        user_details = User.objects.filter(ration_card=ration_card).first()
        
        pass_draw = pass_word()
        user_details.password = make_password(pass_draw)
        user_details.save()
        playsound('updated_db.mp3')
        for key in list(request.session.keys()):
            del request.session[key]
        return HttpResponseRedirect('/login')
           
    return HttpResponse(template.render(context, request))
  
def register(request):
    if request.method == 'POST':
        first_name = request.POST['firstname']
        last_name = request.POST['lastname']
        ration_card = request.POST['rationcardno']
        phone_number = request.POST['phno']
        hint = request.POST['s1']
        hint = make_password(hint)
        secretpass = request.POST["secretpass"]
        secretpass = make_password(secretpass)
        pass_draw = pass_word()
        pass_draw_en = make_password(pass_draw)
        userObj = User.objects.create(first_name = first_name, hint_answer = secretpass, last_name = last_name, phone_number = phone_number, ration_card = ration_card, hint_question = hint, password = pass_draw_en)
        playsound('voice.mp3')
        filename = record_audio_train(ration_card)
        train_model(filename)
        filename = record_audio_test()
        identity = test_model(filename)
        playsound('2_factor_complete.mp3')
        request.session['logged_in'] = True
        request.session['ration_card'] = ration_card
        request.session['first_name'] = first_name
        request.session['phone_number'] = phone_number
        return HttpResponseRedirect('/profile')
    template = loader.get_template('home.html')
    if context['login_req'] == True:
        return render(request, "register.html")
    else:
        return HttpResponse(template.render(context, request))
  
def login(request):
    if request.method == "POST":
        ration_card = request.POST['ration_card']
        
        user_details = User.objects.filter(ration_card=ration_card).first()
        pass_draw = pass_word()
        if user_details and check_password(pass_draw, user_details.password):
            playsound('2_factor.mp3')
            filename = record_audio_test()
            identity = test_model(filename)
            if identity == ration_card:
                request.session['logged_in'] = True
                request.session['ration_card'] = user_details.ration_card
                request.session['first_name'] = user_details.first_name
                request.session['phone_number'] = user_details.phone_number
                return HttpResponseRedirect('/profile')
            else:
                playsound('error.mp3')
        else:
            playsound('error.mp3')
            #return render(request, 'login.html')
    else:
        
        if context['login_req'] == True:
            return render(request, "login.html")
        else:
            return HttpResponse(template.render(context, request))
    
def forgot(request):
    if request.method == 'POST':        
        ration_card = request.POST['rationcardno']
        hint = request.POST['s1']
        secretpass = request.POST["secretpass"] 
        user_details = User.objects.filter(ration_card=ration_card).first()
        if user_details and check_password(secretpass, user_details.hint_answer) and check_password(hint, user_details.hint_question):
            pass_draw = pass_word()
            user_details.password = make_password(pass_draw)
            user_details.save()
            playsound('updated_db.mp3')
            return HttpResponseRedirect('/login')
        else:
            playsound('error_forgot.mp3')
            return HttpResponseRedirect('/forgot')
    template = loader.get_template('home.html')
    if context['login_req'] == True:
        return render(request, "forgot.html")
    else:
        return HttpResponse(template.render(context, request))
    
def profile(request):
    
    if request.method == "POST":
        print("ration")
        print(today)
        
        for key in list(request.session.keys()):
            del request.session[key]
        return HttpResponseRedirect('/')

    template = loader.get_template('home.html')
    if context['login_req'] == True and request.session["logged_in"] == True:
        ration_card = request.session['ration_card']
        
        ration_card = request.session['ration_card']
        ration_details = RationDetails.objects.create(ration_card = ration_card, received_date = today)
        ration_details = RationDetails.objects.filter(ration_card=ration_card).all()
        playsound('updated_db.mp3')
        ration = {
        'ration_details': ration_details,
   	}
   	
        print(ration)
        
        return render(request, "profile.html", ration)
    else:
        return HttpResponse(template.render(context, request))
        
def calculate_delta(array):

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas
    
def extract_features(audio,rate):
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined
    
def record_audio_train(RCNo):
    print("Training recording")
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 5
    device_index = 2
    audio = pyaudio.PyAudio()
    index = 22
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index = index,
                    frames_per_buffer=CHUNK)
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    OUTPUT_FILENAME=RCNo+"-sample"+".wav"
    WAVE_OUTPUT_FILENAME=os.path.join("training_set",OUTPUT_FILENAME)
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()
    return OUTPUT_FILENAME

def record_audio_test():
    print("Testing recording")
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 5
    device_index = 2
    audio = pyaudio.PyAudio()
    
    index = 22
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index = index,
                    frames_per_buffer=CHUNK)
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    OUTPUT_FILENAME="sample.wav"
    WAVE_OUTPUT_FILENAME=os.path.join("testing_set",OUTPUT_FILENAME)
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()
    return OUTPUT_FILENAME

def train_model(OUTPUT_FILENAME):

    source = "./training_set/"+OUTPUT_FILENAME 
    dest = "./trained_models/"
    features = np.asarray(())    
    sr,audio = read(source)
    vector   = extract_features(audio,sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    gmm = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag',n_init = 3)
    gmm.fit(features)
    picklefile = OUTPUT_FILENAME.split("-")[0]+".gmm"
    pickle.dump(gmm,open(dest + picklefile,'wb'))
    os.remove(source)
    features = np.asarray(())


def test_model(OUTPUT_FILENAME):

    source   = "./testing_set/"+OUTPUT_FILENAME  
    modelpath = "./trained_models/"

    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
                  in gmm_files]
    sr,audio = read(source)
    vector   = extract_features(audio,sr)

    log_likelihood = np.zeros(len(models)) 

    for i in range(len(models)):
        gmm    = models[i]  
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    os.remove(source)
    time.sleep(1.0)
    return speakers[winner]
    

model = load_model('./doodle_recognizer.h5')
class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y, std_acc, x_std_meas, y_std_meas):

        # Define sampling time
        self.dt = dt

        # Define the  control input variables
        self.u = np.matrix([[u_x],[u_y]])

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        
        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x[0:2]

    def update(self, z):

        
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x[0:2]

class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image, clear, stop, draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        fingerCount = 0
        gesture = ""
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                handIndex = self.results.multi_hand_landmarks.index(handLms)
                handLabel = self.results.multi_handedness[handIndex].classification[0].label
                handLandmarks = []
                for landmarks in handLms.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount = fingerCount+1
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount = fingerCount+1

                if handLandmarks[8][1] < handLandmarks[6][1]:      
                    fingerCount = fingerCount+1
                if handLandmarks[12][1] < handLandmarks[10][1]:    
                    fingerCount = fingerCount+1
                if handLandmarks[16][1] < handLandmarks[14][1]:  
                    fingerCount = fingerCount+1
                if handLandmarks[20][1] < handLandmarks[18][1]:    
                    fingerCount = fingerCount+1
                if fingerCount == 1 and handLandmarks[8][1] < handLandmarks[6][1]:
                    gesture = "Draw"
                elif fingerCount == 5:
                    gesture = "Erase"
                    cv2.putText(image,'Clearing Drawing', (100,200), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 5, cv2.LINE_AA)
                    clear = True
                    
                elif fingerCount == 0:
                    gesture = "Save"
                    stop = True

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                cv2.putText(image, gesture, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
                
        return image, clear, stop
    
    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                if id == 8:
                    h,w,c = image.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)

                    lmlist.append([cx,cy])
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
                

        return lmlist
        
def pass_word():
    cap = cv2.VideoCapture(0)
    predictions=''
    
    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
    tracker = handTracker()
    canvas = None
    clear = False
    stop = False
    x_track, y_track = 0, 0
    #playsound('start.mp3')
    while True:
        success,image = cap.read()
        image = cv2.flip(image, 1 )
        if canvas is None:
            canvas = np.zeros_like(image)
        if success:
            image, clear, stop= tracker.handsFinder(image, clear, stop)
            lmList = tracker.positionFinder(image)
            if len(lmList) != 0:
                print(lmList[0])
                x_id = lmList[0][0]
                y_id = lmList[0][1]
                cv2.circle(image, (int(x_id), int(y_id)), 10, (0, 191, 255), 2)
                
                (x, y) = KF.predict()
                x = x.tolist()
                y = y.tolist()
                y = y[0][0]
                x = x[0][0]
                
                (x1, y1) = KF.update(lmList[0])
                x1 = x1.tolist()
                y1 = y1.tolist()
                y1 = y1[0][1]
                x1 = x1[0][0]
                
                # Draw a rectangle as the estimated object position
                cv2.rectangle(image, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)
                cv2.putText(image, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
                cv2.putText(image, "Measured Position", (int(lmList[0][0] + 15), int(lmList[0][1] - 15)), 0, 0.5, (0,191,255), 2)
                
                if x_track == 0 and y_track == 0:
                    x_track,y_track= x1, y1
             
                else:
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x_track,y_track),(int(x1),int(y1)), [255,0,0], 4)
         
                # After the line is drawn the new points become the previous points.
                x_track,y_track = int(x1),int(y1)
            else:
                x_track,y_track =0,0
            _ , mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
            foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
            #cv2.imshow('drawing', foreground)
            background = cv2.bitwise_and(image, image, mask = cv2.bitwise_not(mask))
#             image = cv2.add(foreground,background)

            cv2.imshow('image',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if clear == True:
                canvas = None
                #playsound('erase.mp3')
                clear = False
            elif cv2.waitKey(1) & stop == True:
#                 filename = 'hand_doodle3.jpg'
#                 cv2.imwrite(filename, foreground)
#                 img1 = cv2.imread('hand_doodle3.jpg')
                playsound('save.mp3')
                img=Image.fromarray(foreground.astype(np.uint8))
                class_names=['t-shirt', 'book', 'door', 'axe', 'banana', 'donut', 'belt', 'eyeglasses', 'butterfly', 'alarm clock', 'lollipop', 'cell phone', 'scissors', 'bucket', 'basketball', 'bed', 'airplane', 'ceiling fan', 'backpack', 'apple', 'baseball bat', 'chair', 'candle', 'arm', 'bandage', 'birthday cake']
                class_names=sorted(class_names)
                height, width = img.size
                left = 152
                top = 132
                width = 246
                height = 166
                box = (left, top, left+width, top+height)

                #resizing the image to find spaces better
                cropped = img.crop(box)
                #image = cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
                #cv2.imshow("Frame", img)
                img=cropped.resize((64, 64), Image.ANTIALIAS)
                black = (0,0,0)
                white = (255,255,255)
                threshold = (200,200,200)
                # Open input image in grayscale mode and get its pixels.
                pixels = img.getdata()
                newPixels = []

                # Compare each pixel 
                for pixel in pixels:
                    if pixel < threshold:
                        newPixels.append(black)
                    else:
                        newPixels.append(white)

                # Create and save new image.
                newImg = Image.new("RGB",img.size)
                newImg.putdata(newPixels)
                newImg = cv2.cvtColor(np.float32(newImg), cv2.COLOR_BGR2GRAY).reshape(64,64,1)

                   #plt.imshow(newImg)
                   #newImg.save("imagedoodle_cropped.png", quality=100)
                plot=np.asarray(newImg).reshape(64,64,1)
                plot=plot.reshape(1,64,64,1)
                plot /= 255.0
                _,idx=np.unique(plot, axis=0,return_index=True)
                plot=plot[np.sort(idx)]
                pred=model.predict(plot)
                print(pred)
                
                for p in pred:
                    predictions=class_names[np.argmax(p)]
                    
                break
                
    cap.release()
    cv2.destroyAllWindows()
    return predictions


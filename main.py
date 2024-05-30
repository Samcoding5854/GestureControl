import cv2
import time
import subprocess
from HandInfoDetector import HandInfo
from Utils import EmotionDetector
from cvzone.HandTrackingModule import HandDetector
import webbrowser


############################################################


cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

tipIds = [4, 8, 12, 16, 20] #Finger Tips Id for Mediapipe

detectorInd = HandDetector(detectionCon=0.8, maxHands=2)

link_opened = False
cTime = 0
start_time = None
tracking_duration = 60
current_emotion = None
pTime = 0
emotion_actions = {
    "happy": "##INSERT THE PLAYLIST LINK##",
    "sad": "##INSERT THE PLAYLIST LINK##",
    "angry": "##INSERT THE PLAYLIST LINK##",
    "neutral": "##INSERT THE PLAYLIST LINK##",
    "shocked": "##INSERT THE PLAYLIST LINK##"
}

#############################################################

def convert_annotations(current_annotations):
    # Extract the current lmList and add indices
    lmList_with_index = [[i, coord[0], coord[1]] for i, coord in enumerate(current_annotations['lmList'])]

    # Create the converted annotations dictionary
    converted_annotations = {
        'lmList': lmList_with_index,
        'bbox': current_annotations['bbox'],
        'center': current_annotations['center'],
        'type': current_annotations['type']
    }

    return converted_annotations


while True:
    success, img = cap.read()
    emotion = EmotionDetector(img) # Emotion Detector 
    print(emotion)
    hands, img = detectorInd.findHands(img)

    if emotion in emotion_actions:
        if current_emotion != emotion:
            current_emotion = emotion
            start_time = time.time()

        elapsed_time = time.time() - start_time
        if elapsed_time >= tracking_duration and not link_opened:
            action = emotion_actions[current_emotion]
            webbrowser.open(action)  # Open the URL in the default web browser
            link_opened = True
            current_emotion = None

    else:
        current_emotion = None
        start_time = None

    numberOfHands = HandInfo(hands)

    if len(hands) != 0:
        print(f' Number Of Hands: {numberOfHands}')
        ### Play Pause Activation Code ###
        if numberOfHands == '1':
            fingers = []
            hand1 = hands[0]
            print("hand1: ", hand1)
            hand1 = convert_annotations(hand1)
            lmlist = hand1["lmList"]
            print("lmlist: ", lmlist)



            ### Counting the Number Displayed on Hand ###

            if lmlist[tipIds[0]][1] > lmlist[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if lmlist[tipIds[id]][2] < lmlist[tipIds[id] - 2][2]:  # Tips number's Y-Position if less than the 2nd point of the same finger, it's closed and vice-versa
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)

            print(f'Number of Fingers : {totalFingers}')

            ### Running Commands for next / previous / play / pause ###
            if totalFingers == 1:
                if fingers == [0, 0, 0, 0, 1]:
                    subprocess.Popen('nircmd sendkeypress mediaplaypause', shell=True)  # Play/Pause Track
                    print("Play/Pause Track")

                elif fingers == [1, 0, 0, 0, 0]:
                    subprocess.Popen('nircmd sendkeypress prevtrack', shell=True)  # Previous Track
                    print("Previous Track Played")

            elif totalFingers == 0:
                subprocess.Popen('nircmd sendkeypress mediaplaypause', shell=True)  # Pause
                print("Paused")

            elif totalFingers == 5:
                subprocess.Popen('nircmd sendkeypress play', shell=True)  # Play
                print("Play")

        ### Showing the FPS ###
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("IMAGE", img)
    cv2.waitKey(1)

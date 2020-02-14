#------------------------------------------------------------
# SEGMENT, RECOGNIZE and COUNT fingers from a video sequence
#------------------------------------------------------------

# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import tkinter as Tk
import tkinter.messagebox

# messages to disp
Win_msg=["Looks like I Win Again", "Do You Even Know The Rules", "Today is Not Your Day",
         "Stop Crying, Man Up!", "You Came To NYU for This?", "Machine Will rise and Take Over Soon!!" ]

Draw_msg=["You Are a Worthy Opponent", "Ditto, Its A Match", "We Think Alike",
            "Stop Copying ME", "Humans Are Match For AI After All", "I Know Your Next Move" ]

Lose_msg=["GHz In Processing Speed, For Nothing", "You Won the Battle, Not War", "Rub Off that Stupid Smile",
          "Next Move And You Are Dead", "WOW!! Even the Worst Human can Defeat Me", "Sure Add This To Your CV"  ]

# global variables
bg = None
gesture = ""

root = Tk.Tk()
g_User = Tk.StringVar()
g_CPU = Tk.StringVar()
result = Tk.StringVar()
wincount = Tk.DoubleVar()
losscount = Tk.DoubleVar()
drawcount = Tk.DoubleVar()
msg_disp = Tk.StringVar()
#def fun1():
#	Tk.messagebox.showinfo('Message box title', 'Thank You for playing!')
#	root.quit()
#w1 = Tk.Canvas(root, width=200, height=100)
#w2 = Tk.Canvas(root, width=200, height=100)
#w3 = Tk.Canvas(root, width=200, height=100)
#w1.pack()
#w2.pack()
#w3.pack()
#l1 = Tk.Label(root, text = 'Wins')
#l2 = Tk.Label(root, text = 'Loss')
#l3 = Tk.Label(root, text = 'Draws')
#l4 = Tk.Label(root, text = 'SCOREBOARD')
#B1 = Tk.Button(root, text = 'Quit', command = fun1)
#B1.pack()

#w1.create_rectangle(50, 30, 150, 75, fill="white")
#w2.create_rectangle(50, 30, 150, 75, fill="white")
#w3.create_rectangle(50, 30, 150, 75, fill="white")
#w1.create_text(200 / 2,100 / 2,text= str(wincount))
#w2.create_text(200 / 2,100 / 2,text= str(losscount))
#w3.create_text(200 / 2,100 / 2,text= str(drawcount))
#l1.pack()
#l1.place(x=5,y=40)
#l2.pack()
#l2.place(x=5,y=140)
#l3.pack()
#l3.place(x=5,y=250)
#l4.pack()
#l4.place(x= 60, y= 5)

L1 = Tk.Label(root, text = "CPU:")
L2 = Tk.Label(root, text = "YOU:")
L3 = Tk.Label(root, textvariable = g_CPU)
L4 = Tk.Label(root, textvariable = g_User)
L5 = Tk.Label(root, textvariable = result)
L6 = Tk.Label(root, textvariable = str(wincount))
L7 = Tk.Label(root, textvariable = str(drawcount))
L8 = Tk.Label(root, textvariable = str(losscount))
L9 = Tk.Label(root, textvariable = msg_disp)

L1.pack(side = Tk.LEFT)
L3.pack(side = Tk.LEFT)
L4.pack(side = Tk.RIGHT)
L2.pack(side = Tk.RIGHT)
L5.pack()
L9.pack(side = Tk.BOTTOM)
L7.pack(side = Tk.BOTTOM)
L8.pack(side = Tk.BOTTOM)
L6.pack(side = Tk.BOTTOM)


#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)


#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    cnts, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.10)) > (y + h)) and ((circumference * 0.10) > c.shape[0]):
            count += 1

    return count

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 70, 480, 285, 695

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    #Avg for fingers val
    k = 0
    n = 10
    buffer = [0]*n
    output_avg = 0
    Random_gesture = np.random.randint(1, 3, 1) # 1 rock 2 paper 3 scissor 
    # keep looping, until interrupted
    wcount = 0
    lcount = 0
    dcount = 0
    counted = False
    wait = False
    loop_enter = 0
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                fingers = count(thresholded, segmented)
                # print(fingers)
                buffer[k] = fingers
                k=k+1
                if k == n  :
                    gesture = ""
                    k=0
                    output_avg = np.average(buffer)
                    loop_enter = loop_enter + 1
                   


                    if output_avg >=3.5 :
                            
                        gesture = "PAPER" +" " + str(output_avg)
                                                
                        if (Random_gesture == 1):
                            gesture = gesture + "  YOU WIN CPU CHOSE ROCK"     
                            
                        elif (Random_gesture == 2):
                            gesture = gesture +  "  DRAW CPU CHOSE PAPER" 
                            
                        else:
                            gesture = gesture + "  YOU LOSE CPU CHOSE SCISSOR"
                               
                               
                            
                    elif output_avg >= 1.8 :
                        
                        gesture = "SCISSOR"+" " + str(output_avg)
                        
                        if (Random_gesture == 2):
                            gesture = gesture +  "  YOU WIN CPU CHOSE PAPER"
                            
                        elif (Random_gesture == 3):
                            gesture = gesture +  "  DRAW CPU CHOSE SCISSOR"
                            
                        else:
                            gesture = gesture + "  YOU LOSE CPU CHOSE ROCK"
                              
                                
                    else:
                        gesture = "ROCK"+" " + str(output_avg)
                        
                        if (Random_gesture == 3):
                            gesture = gesture +  "  YOU WIN CPU CHOSE SCISSORS"
                            
                        elif (Random_gesture == 1):
                            gesture = gesture +  "  DRAW CPU CHOSE ROCK"
                            
                        else:
                            gesture = gesture + "  YOU LOSE CPU CHOSE PAPER"
                              
                    
                    if ((counted==False) and (loop_enter == 5)): 
                        loop_enter=0  

                        if Random_gesture == 1:
                          g_CPU.set('ROCK')                
                        elif Random_gesture == 2:
                          g_CPU.set('PAPER')              
                        else:
                          g_CPU.set('SCISSORS')

                        print("entered loop")              
                        if output_avg >=3.5 :
                            counted = True
                            gesture = "PAPER" +" " + str(output_avg)
                            g_User.set('PAPER')                       
                            if (Random_gesture == 1):
                               gesture = gesture + "  YOU WIN CPU CHOSE ROCK"     
                               result.set('YOU WON')
                               msg_disp.set(Lose_msg[np.asscalar(np.random.randint(0,6,1))])
                               wcount = wcount + 1
                               wincount.set(wcount)
                            elif (Random_gesture == 2):
                               gesture = gesture +  "  DRAW CPU CHOSE PAPER" 
                               result.set('DRAW') 
                               msg_disp.set(Draw_msg[np.asscalar(np.random.randint(0,6,1))])
                               dcount = dcount + 1
                               drawcount.set(dcount)
                            else:
                               gesture = gesture + "  YOU LOSE CPU CHOSE SCISSOR"
                               result.set('YOU LOSE')
                               msg_disp.set(Win_msg[np.asscalar(np.random.randint(0,6,1))])
                               lcount = lcount + 1
                               losscount.set(lcount)
                               
                        elif output_avg >= 1.8 :
                            counted = True
                            gesture = "SCISSOR"+" " + str(output_avg)
                            g_User.set('SCISSOR')
                            if (Random_gesture == 2):
                               gesture = gesture +  "  YOU WIN CPU CHOSE PAPER"
                               result.set('YOU WON')
                               msg_disp.set(Lose_msg[np.asscalar(np.random.randint(0,6,1))])
                               wcount = wcount + 1
                               wincount.set(wcount)
                            elif (Random_gesture == 3):
                               gesture = gesture +  "  DRAW CPU CHOSE SCISSOR"
                               result.set('DRAW')
                               msg_disp.set(Draw_msg[np.asscalar(np.random.randint(0,6,1))])
                               dcount = dcount + 1
                               drawcount.set(dcount)
                            else:
                               gesture = gesture + "  YOU LOSE CPU CHOSE ROCK"
                               result.set('YOU LOSE')
                               msg_disp.set(Win_msg[np.asscalar(np.random.randint(0,6,1))])
                               lcount = lcount + 1
                               losscount.set(lcount)
                                
                        else:
                            gesture = "ROCK"+" " + str(output_avg)
                            g_User.set('ROCK')
                            counted = True 
                            if (Random_gesture == 3):
                               gesture = gesture +  "  YOU WIN CPU CHOSE SCISSORS"
                               result.set('YOU WON')
                               msg_disp.set(Lose_msg[np.asscalar(np.random.randint(0,6,1))])
                               wcount = wcount + 1
                               wincount.set(wcount)
                            elif (Random_gesture == 1):
                               gesture = gesture +  "  DRAW CPU CHOSE ROCK"
                               result.set('DRAW')
                               msg_disp.set(Draw_msg[np.asscalar(np.random.randint(0,6,1))])
                               dcount = dcount + 1
                               drawcount.set(dcount)
                            else:
                               gesture = gesture + "  YOU LOSE CPU CHOSE PAPER"
                               result.set('YOU LOSE')
                               msg_disp.set(Win_msg[np.asscalar(np.random.randint(0,6,1))])
                               lcount = lcount + 1
                               losscount.set(lcount)
                            
                # print(buffer)
                cv2.putText(clone, gesture, (00, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)
            else:
                cv2.putText(clone, "ARE YOU READY FOR THE BEST GAME EVER ?", (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
                buffer = [0]*n
                Random_gesture = np.random.randint(1, 4, 1)
                gesture=""
                g_User.set('')
                g_CPU.set('')
                result.set('') 
                counted = False
                loop_enter=0
                msg_disp.set("")
                

        
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        elif keypress == ord("c"):
            num_frames = 0
            Random_gesture = np.random.randint(1, 3, 1)
            counted = False
            loop_enter=0
            
        root.update()

# free up memory
camera.release()
cv2.destroyAllWindows()
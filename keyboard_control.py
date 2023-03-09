
from djitellopy import tello
import KeyPressModule as kp
import time
import cv2
from PIL import Image
import numpy as np
from yolo import YOLO

yolo = YOLO()
kp.init()
me = tello.Tello()
me.connect()
initial_battery = me.get_battery()
global img
me.streamon()

# mode 1: normal fly mode
# mode 2: capture mode, capture images consecutively
mode = 1




def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if kp.getKey("LEFT"):
        lr = -speed
    elif kp.getKey("RIGHT"):
        lr = speed
    if kp.getKey("UP"):
        fb = speed
    elif kp.getKey("DOWN"):
        fb = -speed
    if kp.getKey("w"):
        ud = speed
    elif kp.getKey("s"):
        ud = -speed
    if kp.getKey("a"):
        yv = -speed
    elif kp.getKey("d"):
        yv = speed
    if kp.getKey("q"):
        me.land()
        time.sleep(3)
    if kp.getKey("e"):
        me.takeoff()
    if kp.getKey("p"):
        show_battery()
        return False
    if kp.getKey("z"):
        cv2.imwrite(f'dataset/dataset_aphids/{time.time()}.jpg', img)
        time.sleep(0.3)


    return [lr, fb, ud, yv]


def show_battery():
    print("===========================================")
    print("||   Drone initial battery:   " + str(initial_battery) + "%   ||")
    print("===========================================")
    print("||   Drone battery remaining: " + str(me.get_battery()) + "%   ||")
    print("===========================================")
    print("||   Drone battery used:      " +
          str(initial_battery - me.get_battery()) + "%   ||")
    print("===========================================")


def cv_detect(frame, is_first_frame):
    t1 = time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to Image
    frame = Image.fromarray(np.uint8(frame))
    # Detect
    frame = np.array(yolo.detect_image(frame, is_first_frame = is_first_frame, crop = False, count = True))
    # RGBtoBGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame




if mode == 1:
    fps = 0.0
    is_first_frame = True
    while True:
        t1 = time.time()
        vals = getKeyboardInput()
        if (vals == False):
            me.streamoff()
            cv2.destroyAllWindows()
            break
        else:
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
            img = me.get_frame_read().frame
            img = cv_detect(img, is_first_frame)
            is_first_frame = False
            #img = cv2.resize(img, (360, 240))
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            img = cv2.putText(img, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Drone Camera", img)
            cv2.waitKey(1)

        time.sleep(0.005)

elif mode == 2:
    while True:
        me.send_rc_control(0, 0, 0, 0)
        img = me.get_frame_read().frame
        #img = cv_detect(img)
        #img = cv2.resize(img, (360, 240))
        cv2.imshow("Drone Camera", img)
        cv2.imwrite(f'dataset/dataset_aphids/{time.time()}.jpg', img)
        time.sleep(1)
        cv2.waitKey(1)
        

        time.sleep(0.05)

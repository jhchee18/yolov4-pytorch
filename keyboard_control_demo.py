
from djitellopy import tello
import KeyPressModule as kp
import time
import cv2
from PIL import Image
import numpy as np
from yolo import YOLO
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

yolo = YOLO()
kp.init()
me = tello.Tello()
me.connect()
initial_battery = me.get_battery()
global img
me.streamon()

# mode 1: normal fly mode
# mode 2: capture mode, capture images consecutively
# mode 3: able to follow saved path by pressing "r" key
mode = 3

# mode save_to_demo, True --> save to demo.txt
save_to_demo = False
demo_file_path = "demo.txt"

#global recorded_path
recorded_path = ""

# Create a figure and add a 3D axis to it
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Set the axis limits to be -1 and 1, with an interval of 1
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xticks(np.arange(-100, 100, 20))
ax.set_yticks(np.arange(-100, 100, 20))
ax.set_zticks(np.arange(-100, 100, 20))

# Create an array to store the x, y, and z coordinates of the points
n_points = 100
xs = np.zeros(n_points)
ys = np.zeros(n_points)
zs = np.zeros(n_points)

# Use the scatter method to create a scatter plot of the points
points = ax.scatter(xs, ys, zs)

ax.set_xlabel('X-Axis (cm)')
ax.set_ylabel('Y-Axis (cm)')
ax.set_zlabel('Z-Axis (cm)')


### PARAMETERS ###
fSpeed = 117 / 10 # Forward Speed in cm/s
aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)
interval = 0.25
dInterval = fSpeed * interval
aInterval = aSpeed * interval

x, y = 500, 500
a = 0
yaw = 0





def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    
    global xs, ys, zs
    global x, y, yaw, a
    d = 0

    global recorded_path
        
    # Shift the x, y, and z coordinates of the points by one index
    xs[:-1] = xs[1:]
    ys[:-1] = ys[1:]
    zs[:-1] = zs[1:]
    

    if kp.getKey("LEFT"): # Drone translates leftwards
        lr = -speed
        d = dInterval
        a = -180
    elif kp.getKey("RIGHT"): # Drone translates rightwards
        lr = speed
        d = -dInterval
        a = 180
    if kp.getKey("UP"): # Drone translates forwards
        fb = speed
        d = -dInterval
        a = -90
    elif kp.getKey("DOWN"): # Drone translates backwards
        fb = -speed
        d = dInterval
        a = 270
    if kp.getKey("w"): # Drone translates upwards
        ud = speed
        zs[-1] += dInterval
    elif kp.getKey("s"): # Drone translates downwards
        ud = -speed
        zs[-1] += -dInterval
    if kp.getKey("a"): # Drone rotates anticlockwise
        yv = -speed
        yaw += aInterval
    elif kp.getKey("d"): # Drone rotates clockwise
        yv = speed
        yaw -= aInterval
    if kp.getKey("q"): # Drone lands
        me.land()
        time.sleep(3)
    if kp.getKey("e"): # Drone takes off
        me.takeoff()
    if kp.getKey("p"): # Terminate after landing
        if save_to_demo:
            with open(demo_file_path, 'w') as f:
                f.write(recorded_path)
                f.close()
        show_battery()
        return False
    if kp.getKey("z"): # Capture photo
        cv2.imwrite(f'dataset/dataset_aphids/{time.time()}.jpg', img)
        time.sleep(0.3)
    if kp.getKey("r"): # Follow fly path
        fly_path = read_fly_path(demo_file_path)
        return fly_path


    time.sleep(0.01)
    a += yaw
    xs[-1] += int(d * math.cos(math.radians(a)))
    ys[-1] += int(d * math.sin(math.radians(a)))

    points._offsets3d = (xs, ys, zs)

    recorded_path += str([lr, fb, ud, yv]) + "\n"

    return [lr, fb, ud, yv]


def read_fly_path(demo_file_path):
    fly_path = []

    with open(demo_file_path, "r") as file:
        for line in file:
            line = line.strip()
            
            # Convert the line into a list using eval() function
            command = eval(line)
            
            fly_path.append(command)

    file.close()
    return fly_path


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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to Image
    frame = Image.fromarray(np.uint8(frame))
    # Detect
    frame = np.array(yolo.detect_image(frame, is_first_frame = is_first_frame, crop = False, count = True))
    # RGBtoBGR to fulfill opencv format
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
            plt.pause(0.01)
            plt.draw()
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

elif mode == 3:
    fps = 0.0
    is_first_frame = True


    while True:
        t1 = time.time()
        vals = getKeyboardInput()
        if (vals == False):
            me.streamoff()
            cv2.destroyAllWindows()
            break

        # Check if the list is a nested list
        elif all(isinstance(elem, list) for elem in vals):
            for command in vals:
                plt.pause(0.01)
                plt.draw()
                me.send_rc_control(command[0], command[1], command[2], command[3])
                img = me.get_frame_read().frame
                img = cv_detect(img, is_first_frame)
                is_first_frame = False
                #img = cv2.resize(img, (360, 240))
                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                print("fps= %.2f"%(fps))
                img = cv2.putText(img, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Drone Camera", img)
                cv2.waitKey(1)


        else:
            plt.pause(0.01)
            plt.draw()
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

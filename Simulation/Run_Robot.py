import dbus
import dbus.mainloop.glib
import urllib.request as req
import time

import threading

from socket import *
from select import *
import sys

from gi.repository import GObject
from optparse import OptionParser

proxSensorsVal=[0,0,0,0,0]
proxGroundVal=[0,0]
sensor = 0
Velocity = 0
Orientation = 0
Distance_Target = 2500
check_start = 0
check_bottom = 0
Start_Time = time.time()


# set socekt communication
HOST = '192.168.0.4'
PORT = 10000
BUFSIZE = 1024
ADDR = (HOST, PORT)
clientSocket = socket(AF_INET,SOCK_STREAM)
clientSocket.connect(ADDR)
print('Succeed Socket Connection')


def Socket_Receive(sock) :
    global Velocity
    global Orientation

    while True :
        revData = sock.recv(1024)
        revData = revData.decode()
        print(revData)
        data = revData.split()
        if (len(data) == 4) :
            Velocity = float(data[2])
            Orientation = float(data[3])
        else :
            Velocity = float(data[0])
            Orientation = float(data[1])




def Follower():


    global Start_Time
    global check_start
    global Velocity
    global check_bottom
    global Orientation
    global WheelSpeed
    network.GetVariable("thymio-II", "prox.horizontal",reply_handler=get_variables_reply,error_handler=get_variables_error)
    network.GetVariable("thymio-II", "prox.ground.ambiant",reply_handler=get_variables_reply2,error_handler=get_variables_error)

    set_Time = time.time() - Start_Time

    if int(set_Time)<2 :
        Velocity = 0
        Orientation = 0


    WheelSpeed = Speed(Velocity, Orientation)

    network.SetVariable("thymio-II", "motor.left.target", [WheelSpeed['left']])
    network.SetVariable("thymio-II", "motor.right.target", [WheelSpeed['right']])
    print("Time = "+str(set_Time))
    print("Left ="+str(WheelSpeed['left']) +"//Right ="+str(WheelSpeed['right']))
    print("V and O : " + str(Velocity)+" " + str(Orientation))
    print("-----------------------------")
    return True


def Speed(v,o) :
    wheelspeed = {'left' :0, 'right' : 0}
    R = 2.2 ## cm
    L = 9.5 ## cm (distance from center of leftwheel to center of rightwheel)
            ##  // 11 cm (from end of left wheel to end of rightwheel)

    if v< 0 :
        v = 0

    o = o * (3.14/180)

    wheelspeed['left'] = v-9.5*o
    wheelspeed['right'] = v+9.5*o

    ## 500 = 14 cm/s  Therefore we need conduct 35 to each velocity values
    wheelspeed['left'] = wheelspeed['left']*35
    wheelspeed['right'] = wheelspeed['right']*35

    return wheelspeed

def get_variables_reply2(r):
    global proxGroundVal
    proxGroundVal=r

def get_variables_reply(r):
    global proxSensorsVal
    proxSensorsVal=r

def get_variables_error(e):
    print('error:')
    print(str(e))
    loop.quit()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--system", action="store_true", dest="system", default=False,help="use the system bus instead of the session bus")

    (options, args) = parser.parse_args()

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    if options.system:
        bus = dbus.SystemBus()
    else:
        bus = dbus.SessionBus()

    #Create Aseba network
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')

    #print in the terminal the name of each Aseba NOde
    print( network.GetNodesList())




    #GObject loop
    Start_Time = time.time()
    print('receive thread start')
    receiver = threading.Thread(target=Socket_Receive, args=(clientSocket,))
    receiver.start()

    print( 'starting loop')
    loop = GObject.MainLoop()
    #call the callback of Braitenberg algorithm
    handle = GObject.timeout_add (100, Follower) #every 0.1 sec
    loop.run()

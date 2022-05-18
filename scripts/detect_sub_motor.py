#!/usr/bin/env python3
from sys import breakpointhook
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Int32
motor_vel0 = 0
motor_vel1 = 50
motor_vel2 = 100
motor_vel3 = 250

def camera_callback(data):
    global position
    position = data.data
    rospy.loginfo("Position is %d", position)
    
    motorL_pub = rospy.Publisher('motorL_chatter', Int32, queue_size=100)
    motorR_pub = rospy.Publisher('motorR_chatter', Int32, queue_size=100)
    if position == -2:
        velocity_L = motor_vel1
        velocity_R = motor_vel3
        motorL_pub.publish(velocity_L)
        motorR_pub.publish(velocity_R)
        print('go to the left')
    elif position == -1:
        velocity_L = motor_vel2
        velocity_R = motor_vel3
        motorL_pub.publish(velocity_L)
        motorR_pub.publish(velocity_R)
        print('go to the left')
    elif position == 0:
        velocity_L = motor_vel3
        velocity_R = motor_vel3
        motorL_pub.publish(velocity_L)
        motorR_pub.publish(velocity_R)
        print('go to the straight')
    elif position == 1:
        velocity_L = motor_vel3
        velocity_R = motor_vel2
        motorL_pub.publish(velocity_L)
        motorR_pub.publish(velocity_R)
        print('go to the right')
    elif position == 2:
        velocity_L = motor_vel3
        velocity_R = motor_vel2
        motorL_pub.publish(velocity_L)
        motorR_pub.publish(velocity_R)
        print('go to the right')
    else:
        velocity_L = motor_vel0
        velocity_R = motor_vel0
        motorL_pub.publish(velocity_L)
        motorR_pub.publish(velocity_R)
        print('stop')
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    
    rospy.init_node('listener', anonymous=False)
    rospy.Subscriber('camera_chatter', Int32, camera_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
if __name__ == '__main__':
    listener()

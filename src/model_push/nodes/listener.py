#!/usr/bin/env python
import roslib
import sys
import math

import rospy
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetLinkState

def gms_client(model_name,relative_entity_name):
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp1 = gms(model_name,relative_entity_name)
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def gls_client(link_name,reference_frame):
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        gls = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        resp1 = gls(link_name,reference_frame)
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def vector3_mag(v):
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

if __name__ == "__main__":
    model_name = "my_robot"
    left_wheel_name = model_name + "::left_wheel"
    right_wheel_name = model_name + "::right_wheel"
    relative_entity_name = "world"
    while True:
      res = gms_client(model_name,relative_entity_name)
      print "model position:"
      print res.pose.position
      res = gls_client(left_wheel_name,relative_entity_name)
      print "left wheel linear velocity magnitude:"
      print vector3_mag(res.link_state.twist.linear)
      res = gls_client(right_wheel_name,relative_entity_name)
      print "right wheel linear velocity magnitude:"
      print vector3_mag(res.link_state.twist.linear)

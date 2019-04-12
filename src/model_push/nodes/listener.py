#!/usr/bin/env python
import roslib
import sys

import rospy
from gazebo_msgs.srv import GetModelState

def gms_client(model_name,relative_entity_name):
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp1 = gms(model_name,relative_entity_name)
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


if __name__ == "__main__":
    model_name = "ball"
    relative_entity_name = "link"
    while True:
      res = gms_client(model_name,relative_entity_name)
      print "return x position ",res.pose.position

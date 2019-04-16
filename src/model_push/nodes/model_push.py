#!/usr/bin/env python

import math
import roslib

import sys, unittest
import os, os.path, time
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Quaternion, Point, PoseStamped, PoseWithCovariance, TwistWithCovariance, Twist, Vector3, Wrench
from gazebo_msgs.srv import ApplyBodyWrench
import tf.transformations as tft
from numpy import float64

#initialize ros node
rospy.init_node("model_push", anonymous=True)
rospy.wait_for_service('/gazebo/apply_body_wrench')
apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

wrench = Wrench()
wrench.force.x = 3
wrench.force.y = 0
wrench.force.z = 0
wrench.torque.x = 0
wrench.torque.y = 0
wrench.torque.z = 0

try:
    print apply_body_wrench(body_name = "ball::link",
                 reference_frame = "ball::link",
                 wrench = wrench,
                 duration = rospy.Duration(.5))
    print "done"
except rospy.ServiceException as e:
    print e

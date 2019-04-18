#!/usr/bin/env python
import os 
import roslib
import sys
import math
import csv
import rospy
from gazebo_msgs.srv import GetPhysicsProperties
from gazebo_msgs.srv import SetPhysicsProperties
from gazebo_msgs.srv import GetWorldProperties
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import Point
#from std_msgs.msg import String

def gpp_client():
    rospy.wait_for_service('/gazebo/get_physics_properties')
    try:
        gpp = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        resp1 = gpp()
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def spp_client(time_step, max_update_rate, gravity, ode_config):
    rospy.wait_for_service('/gazebo/set_physics_properties')
    try:
        spp = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        resp1 = spp(time_step, max_update_rate, gravity, ode_config)
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def gwp_client():
    rospy.wait_for_service('/gazebo/get_world_properties')
    try:
        gwp = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        resp1 = gwp()
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

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
    print(os.getcwd())

    model_name = "my_robot"
    left_wheel_name = model_name + "::left_wheel"
    right_wheel_name = model_name + "::right_wheel"
    world_name = "world"

    forces = [1]

    data = []

    pub = rospy.Publisher('force_cmd', Point)
    rospy.init_node('siso_client', anonymous=True)
    rate = rospy.Rate(10)

    moving_threshold = 0.01

    physics_res = gpp_client()
    #spp_client(0.000001, physics_res.max_update_rate, physics_res.gravity, physics_res.ode_config)

    for force in forces:
        print("Running car with force %d..." % force) 

        msg = Point()
        msg.x = force
        msg.y = force

        while True:
            world_res = gwp_client()
            sim_time = world_res.sim_time

            model_res = gms_client(model_name,world_name)
            model_pos = model_res.pose.position

            left_wheel_res = gls_client(left_wheel_name,world_name)
            right_wheel_res = gls_client(right_wheel_name,world_name)
            left_wheel_vel = left_wheel_res.link_state.twist.angular
            right_wheel_vel = right_wheel_res.link_state.twist.angular

            data_entry = [sim_time,
                          left_wheel_vel.x,
                          left_wheel_vel.y,
                          left_wheel_vel.z,
                          right_wheel_vel.x,
                          right_wheel_vel.y,
                          right_wheel_vel.z,
                          model_pos.x,
                          model_pos.y,
                          model_pos.z]

            #print(','.join([str(x) for x in data_entry]))

            data.append(data_entry)

            if sim_time <= 1:
                pub.publish(msg)

            if (sim_time > 1 and vector3_mag(left_wheel_vel) < moving_threshold and vector3_mag(right_wheel_vel) < moving_threshold) or (sim_time > 180):
                print("Finished")
                break; 

            rate.sleep()
        
        with open('data.csv', 'w+') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)

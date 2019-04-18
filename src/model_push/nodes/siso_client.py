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
import math
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

<<<<<<< HEAD
def truncate(num, digits):
  stepper = pow(10.0, digits)
  return math.trunc(num*stepper)/stepper

=======
>>>>>>> 83d8da95725210cbb99431509990829b766e6d5b
if __name__ == "__main__":
    print(os.getcwd())

    model_name = "my_robot"
    left_wheel_name = model_name + "::left_wheel"
    right_wheel_name = model_name + "::right_wheel"
    world_name = "world"

<<<<<<< HEAD
    forces = [20]
    pub_num_times = [1]

    data = []
    # create a ros node for the robot car plugin
    pub = rospy.Publisher('robot_car_node/force_cmd', Point, queue_size=10)
    rospy.init_node('siso_client')
=======
    forces = [1]

    data = []

    pub = rospy.Publisher('force_cmd', Point)
    rospy.init_node('siso_client', anonymous=True)
>>>>>>> 83d8da95725210cbb99431509990829b766e6d5b
    rate = rospy.Rate(10)

    moving_threshold = 0.01

    physics_res = gpp_client()
    #spp_client(0.000001, physics_res.max_update_rate, physics_res.gravity, physics_res.ode_config)

<<<<<<< HEAD

    for idx, force in enumerate(forces):
=======
    for force in forces:
>>>>>>> 83d8da95725210cbb99431509990829b766e6d5b
        print("Running car with force %d..." % force) 

        msg = Point()
        msg.x = force
        msg.y = force

<<<<<<< HEAD
        counter = 0     
        # flag to check whether the robot car has moved 
        moved = False
        rospy.sleep(15.)
        while True:
            if counter < pub_num_times[idx]:
              pub.publish(msg)
              counter += 1
=======
        while True:
>>>>>>> 83d8da95725210cbb99431509990829b766e6d5b
            world_res = gwp_client()
            sim_time = world_res.sim_time

            model_res = gms_client(model_name,world_name)
            model_pos = model_res.pose.position

            left_wheel_res = gls_client(left_wheel_name,world_name)
            right_wheel_res = gls_client(right_wheel_name,world_name)
            left_wheel_vel = left_wheel_res.link_state.twist.angular
            right_wheel_vel = right_wheel_res.link_state.twist.angular

            data_entry = [sim_time,
<<<<<<< HEAD
                          truncate(left_wheel_vel.x, 6),
                          truncate(left_wheel_vel.y, 6),
                          truncate(left_wheel_vel.z, 6),
                          truncate(right_wheel_vel.x, 6),
                          truncate(right_wheel_vel.y, 6),
                          truncate(right_wheel_vel.z, 6),
                          truncate(model_pos.x, 6),
                          truncate(model_pos.y, 6),
                          truncate(model_pos.z, 6)]

            data.append(data_entry)

            
            if vector3_mag(left_wheel_vel) < moving_threshold and vector3_mag(right_wheel_vel) < moving_threshold and moved:
                print("Finished")
                break
            else:
              moved = True
            
            rate.sleep()
        
        with open('/home/chiliwei/siso_bot/src/model_push/data/data.csv', 'w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['left_wheel_vel_x', 'left_wheel_vel_y', 'left_wheel_vel_z', 'right_wheel_vel_x', 'right_wheel_vel_y', 'right_wheel_vel_z', 'model_pos_x', 'model_pos_y', 'model_pos_z'])
            for row in data:
                writer.writerow(row)
        print('wrote to file')
    rospy.spin()
=======
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
>>>>>>> 83d8da95725210cbb99431509990829b766e6d5b

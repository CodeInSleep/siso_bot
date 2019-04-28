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

fname = 'data.csv'

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

def truncate(num, digits):
  stepper = pow(10.0, digits)
  return math.trunc(num*stepper)/stepper

if __name__ == "__main__":

    if not os.path.isdir(os.environ['SISO_DATA_DIR']):
        print('invalid DATA_DIR (set it in ~/sisobot/export_path.sh')
    
    dirpath = os.environ['SISO_DATA_DIR']
    
    model_name = "my_robot"
    left_wheel_name = model_name + "::left_wheel"
    right_wheel_name = model_name + "::right_wheel"
    world_name = "world"

    forces = [20]
    pub_num_times = [1]

    data = []
    # create a ros node for the robot car plugin
    pub = rospy.Publisher('robot_car_node/force_cmd', Point, queue_size=10)
    rospy.init_node('siso_client')
    rate = rospy.Rate(10)

    moving_threshold = 0.01

    physics_res = gpp_client()
    #spp_client(0.000001, physics_res.max_update_rate, physics_res.gravity, physics_res.ode_config)

    for idx, force in enumerate(forces):
        print("Running car with force %d..." % force) 

        msg = Point()
        msg.x = force
        msg.y = force

        counter = 0     
        # flag to check whether the robot car has moved 
        moved = False
        rospy.sleep(15.)
        while True:
            if counter < pub_num_times[idx]:
              pub.publish(msg)
              counter += 1
            world_res = gwp_client()
            sim_time = world_res.sim_time

            model_res = gms_client(model_name,world_name)
            model_pos = model_res.pose.position

            left_wheel_res = gls_client(left_wheel_name,world_name)
            right_wheel_res = gls_client(right_wheel_name,world_name)
            left_wheel_vel = left_wheel_res.link_state.twist.angular
            right_wheel_vel = right_wheel_res.link_state.twist.angular

            data_entry = [sim_time,
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

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)        
        with open(os.path.join(dirpath, fname), 'w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sim_time', 'left_wheel_vel_x', 'left_wheel_vel_y', 'left_wheel_vel_z', 'right_wheel_vel_x', 'right_wheel_vel_y', 'right_wheel_vel_z', 'model_pos_x', 'model_pos_y', 'model_pos_z'])
            for row in data:
                writer.writerow(row)
        print('wrote to file')
    rospy.spin()

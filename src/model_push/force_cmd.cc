#include <gazebo/gazebo_config.h>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <geometry_msgs/Wrench.h>
#include "ros/ros.h"


using namespace std;
int main(int _argc, char **_argv) {
  ros::init(_argc, _argv, "force_cmder");

  ros::NodeHandle n;

  ros::Publisher pub = n.advertise<geometry_msgs::Wrench>("force_cmd", 1);

  ros::Rate loop_rate(10);

  while (ros::ok()) {
    geometry_msgs::Wrench msg;
    char x_mag[100];
    cin >> x_mag;

    cout << "publishing: " << x_mag << endl;
    msg.force.x = atof(x_mag);    
    msg.force.y = 0;
    msg.force.z = 0;
    msg.torque.x = 0;
    msg.torque.y = 0;
    msg.torque.z = 0;

    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}

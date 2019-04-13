#include <gazebo/gazebo_config.h>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include "ros/ros.h"
#include "std_msgs/Float64.h"

using namespace std;
int main(int _argc, char **_argv) {
  ros::init(_argc, _argv, "force_cmder");

  ros::NodeHandle n;

  ros::Publisher pub = n.advertise<std_msgs::Float64>("force_cmd", 100);

  ros::Rate loop_rate(10);

  while (ros::ok()) {
    std_msgs::Float64 msg;
    char x_mag[100];
    cin >> x_mag;

    cout << "publishing: " << x_mag << endl;
    msg.data = atof(x_mag);
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}

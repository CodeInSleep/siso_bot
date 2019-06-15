#include <gazebo/gazebo_config.h>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <geometry_msgs/Point.h>
#include "ros/ros.h"

using namespace std;
int main(int _argc, char **_argv) {
  ros::init(_argc, _argv, "force_cmder");

  ros::NodeHandle n;

  ros::Publisher pub = n.advertise<geometry_msgs::Point>("force_cmd", 1);

  ros::Rate loop_rate(10);

  while (ros::ok()) {
    geometry_msgs::Point msg;
    char input[100];
    cin >> input;
    cout << "publishing: " << input << endl;   
    char *pt;
    int counter = 0;
    pt = strtok(input, ",");

    while (pt != NULL) {
      if (counter == 0)
        msg.x = atof(pt);
      else if (counter == 1)
        msg.y = atof(pt);
      else {
        std::cerr << "too many input arguments, exiting.." << std::endl;
        return 1;
      }
      pt = strtok(NULL, ",");
      counter++;
    }

    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}

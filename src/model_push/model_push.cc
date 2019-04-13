#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include "ros/ros.h"
#include "std_msgs/Float64.h"

using namespace std;

namespace gazebo
{
  class ModelPush : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
    {
      // Store the pointer to the model
      this->model = _parent;

      double x_mag = 0;
      if (_sdf->HasElement("force"))
        x_mag = _sdf->Get<double>("force");
      
      this->setForce(x_mag);
      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      //this->updateConnection = event::Events::ConnectWorldUpdateBegin(
      //    std::bind(&ModelPush::OnUpdate, this, mag_x));
      this->sub = this->n.subscribe("force_cmd", 10, &ModelPush::onForceMsg, this);
      ros::MultiThreadedSpinner spinner(2);
    }

    // Called by the world update start event
    /*public: void OnUpdate(double mag_x)
    {
      // Apply a small linear velocity to the model.
      physics::LinkPtr body = this->model->GetLink("link");
      math::Vector3 forceVec(mag_x, 0, 0);      
      math::Vector3 relVec(-0.5, 0, 0);
      body->AddForceAtRelativePosition(forceVec, relVec);
    }*/

    private: void onForceMsg(const std_msgs::Float64::ConstPtr& msg) {
      double x_mag = msg->data;
      std::ostringstream ss;
      ss << msg->data;
      std::string s(ss.str());
      cout << "received: " << s << endl;
      this->setForce(msg->data);
    }

    private: void setForce(const double &x_mag) {
      // Apply a small linear velocity to the model.
      cout << "setting force." << endl;
      std::ostringstream ss;
      ss << x_mag;
      std::string s(ss.str());
      cout << s << endl;

      physics::LinkPtr body = this->model->GetLink("link");
      math::Vector3 forceVec(x_mag, 0, 0);      
      math::Vector3 relVec(-0.5, 0, 0);
      body->AddForceAtRelativePosition(forceVec, relVec);
    }
    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
    private: ros::NodeHandle n;
    private: ros::Subscriber sub;  
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(ModelPush)
}


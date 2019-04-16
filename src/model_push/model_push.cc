#include <algorithm>
#include <assert.h>
#include <time.h>
#include "model_push.h"
namespace gazebo
{
GZ_REGISTER_MODEL_PLUGIN(GazeboRosForce);

////////////////////////////////////////////////////////////////////////////////
// Constructor
GazeboRosForce::GazeboRosForce()
{
  this->ljf_ = 0;
  this->rjf_ = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Destructor
GazeboRosForce::~GazeboRosForce()
{
  this->update_connection_.reset();

  // Custom Callback Queue
  this->queue_.clear();
  this->queue_.disable();
  this->rosnode_->shutdown();
  this->callback_queue_thread_.join();

  delete this->rosnode_;
}

////////////////////////////////////////////////////////////////////////////////
// Load the controller
void GazeboRosForce::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  this->model_ = _model;
  // Get the world name.
  this->world_ = _model->GetWorld();

  this->chassis_ = _model->GetLink("link");  
  if (!_sdf->HasElement("topicName"))
  {
    ROS_FATAL_NAMED("force", "force plugin missing <topicName>, cannot proceed");
    return;
  }
  else
    this->topic_name_ = _sdf->GetElement("topicName")->Get<std::string>();

  std::cout << "robotNamespace: " << this->robot_namespace_ << ", " <<  
      "subscribing to topic: " << this->topic_name_ << ", " << std::endl;
  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM_NAMED("force", "A ROS node for Gazebo has not been initialized, unable to load plugin. "
      << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  this->robot_namespace_ = "myrobot_ns";
  this->rosnode_ = new ros::NodeHandle(this->robot_namespace_);
  std::cout << "ROS NODE initialized" << std::endl;
  // Custom Callback Queue
  
ros::SubscribeOptions so = ros::SubscribeOptions::create<geometry_msgs::Point>(
    this->topic_name_,1,
    boost::bind( &GazeboRosForce::UpdateObjectForce,this,_1),
    ros::VoidPtr(), &this->queue_);
  this->sub_ = this->rosnode_->subscribe(so);

  // Custom Callback Queue
  this->callback_queue_thread_ = boost::thread( boost::bind( &GazeboRosForce::QueueThread,this ) );


  // New Mechanism for Updating every World Cycle
  // Listen to the update event. This event is broadcast every
  // simulation iteration.
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
      boost::bind(&GazeboRosForce::UpdateChild, this));
}

std::string printFloat(const float &f) {
  std::ostringstream ss;
  ss << f;
  std::string s(ss.str());
  return s;
}
////////////////////////////////////////////////////////////////////////////////
// Update the controller
void GazeboRosForce::UpdateObjectForce(const geometry_msgs::Point::ConstPtr& _msg)
{
  std::cout << "updating object force.." << std::endl;
  std::cout << "left joint: " << printFloat(_msg->x) << ", right joint: " << printFloat(_msg->y) << std::endl; 

  this->ljf_ = _msg->x;
  this->rjf_ = _msg->y;
 
  this->timer = this->rosnode_->createTimer(ros::Duration(1), boost::bind(&GazeboRosForce::ResetForce, this, _1), true);
 

  //ignition::math::Vector3d force(_msg->x, 0, 0);
  //ignition::math::Vector3d rel(-0.2, 0.2, 0.1);
 
  /* 
  int running = 1;
  time_t start = time(NULL);
  time_t end;
  while (running) {
    end = time(NULL);
    if (difftime(end, start) > 1)
      running = 0;
    this->chassis_->AddForceAtRelativePosition(force, rel);
  }*/
   //this->r_joint_->SetForce(0, _msg->x);
  //this->l_joint_->SetForce(0, _msg->y);
  // reset forces to 0 after 1 second
   /*
  time_t start, end;
  double elapsed;
  start = time(NULL);
  int running = 1;
  
  this->lock_.lock();
  ignition::math::Vector3d force(_msg->force.x,_msg->force.y,_msg->force.z);
  this->link_->SetForce(force);
  this->lock_.unlock();

  while (running) {
    end = time(NULL);
    elapsed = difftime(end, start);
    if (elapsed >= 1)
      running = 0;
  }

  ignition::math::Vector3d zforce(0, 0, 0);
  this->link_->SetForce(zforce);
*/
}

void GazeboRosForce::ResetForce(const ros::TimerEvent& event) {
  //std::cout << "reseting forces.." << std::endl;
  this->chassis_->ResetPhysicsStates();
  this->ljf_ = 0;
  this->rjf_ = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Update the controller
void GazeboRosForce::UpdateChild()
{
  //double fx = this->wrench_msg_.force.x;
  //double fy = this->wrench_msg_.force.y;
  //double fz = this->wrench_msg_.force.z;
  //std::cout << "fx: " << printFloat(fx) << std::endl;
  //std::cout << "fy: " << printFloat(fy) << std::endl;
  //std::cout << "fz: " << printFloat(fz) << std::endl;

  //ignition::math::Vector3d force(this->ljf_, 0, 0);
  //ignition::math::Vector3d rel(-0.2, 0, 0.1);  
  this->l_joint_->SetForce(2, this->ljf_);
  this->r_joint_->SetForce(2, this->rjf_);
}


// Custom Callback Queue
////////////////////////////////////////////////////////////////////////////////
// custom callback queue thread

void GazeboRosForce::QueueThread()
{
  static const double timeout = 0.05;

  while (this->rosnode_->ok())
  {
    this->queue_.callOne(ros::WallDuration(timeout));
    
  }
}

}


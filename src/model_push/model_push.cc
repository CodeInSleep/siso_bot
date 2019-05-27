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
  this->ljv_ = 0;
  this->rjv_ = 0;
  int argc = 0;
  char **argv = NULL;
  ros::init(argc, argv, "robot_car_node");
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

  // load parameters
  this->robot_namespace_ = "";
  if (_sdf->HasElement("robotNamespace"))
    this->robot_namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>() + "/";

  if (!_sdf->HasElement("leftJoint"))
  {
    ROS_FATAL_NAMED("force", "force plugin missing <leftJoint>, cannot proceed");
    return;
  }
  else
    this->lj_name_ = _sdf->GetElement("leftJoint")->Get<std::string>();

  if (!_sdf->HasElement("rightJoint"))
  {
    ROS_FATAL_NAMED("force", "force plugin missing <rightJoint>, cannot proceed");
    return;
  }
  else
    this->rj_name_ = _sdf->GetElement("rightJoint")->Get<std::string>();

  this->l_joint_ = _model->GetJoint(this->lj_name_);
  if (!this->l_joint_)
  {
    ROS_FATAL_NAMED("force", "gazebo_ros_force plugin error: link named: left joint does not exist\n");
    return;
  }
  this->r_joint_ = _model->GetJoint(this->rj_name_);
  if (!this->r_joint_)
  {
    ROS_FATAL_NAMED("force", "gazebo_ros_force plugin error: link named: right joint does not exist\n");
    return;
  }

  this->chassis_ = _model->GetLink("chassis");
  if (!_sdf->HasElement("topicName"))
  {
    ROS_FATAL_NAMED("force", "force plugin missing <topicName>, cannot proceed");
    return;
  }
  else
    this->topic_name_ = _sdf->GetElement("topicName")->Get<std::string>();

  std::cout << "robotNamespace: " << this->robot_namespace_ << ", " <<  
      "subscribing to topic: " << this->topic_name_ << ", " << 
      "applying to link: " << this->lj_name_ << " and " << this->rj_name_ << std::endl;
  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM_NAMED("force", "A ROS node for Gazebo has not been initialized, unable to load plugin. "
      << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  this->rosnode_ = new ros::NodeHandle(this->robot_namespace_);
  std::cout << "ROS NODE initialized" << std::endl;
  
  this->pub_ = this->rosnode_->advertise<geometry_msgs::Point32>("current_input", 10);
  
  // Custom Callback Queue  
  ros::SubscribeOptions so = ros::SubscribeOptions::create<geometry_msgs::Point32>(
    this->topic_name_,1,
    boost::bind( &GazeboRosForce::UpdateObjectForce,this,_1),
    ros::VoidPtr(), &this->queue_);
  this->sub_ = this->rosnode_->subscribe(so);

  // Custom Callback Queue
  this->callback_queue_thread_ = boost::thread( boost::bind( &GazeboRosForce::QueueThread,this ) );


  // New Mechanism for Updating every World Cycle
  // Listen to the update event. This event is broadcast every
  // simulation iteration.
  // this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
  //    boost::bind(&GazeboRosForce::UpdateChild, this));
}

std::string printFloat(const float &f) {
  std::ostringstream ss;
  ss << f;
  std::string s(ss.str());
  return s;
}
////////////////////////////////////////////////////////////////////////////////
// Update the controller
void GazeboRosForce::UpdateObjectForce(const geometry_msgs::Point32::ConstPtr& _msg)
{
  std::cout << "updating object force.." << std::endl;
  std::cout << "left joint: " << printFloat(_msg->x) << ", right joint: " << printFloat(_msg->y) << std::endl; 

  this->ljv_ = _msg->x;
  this->rjv_ = _msg->y;
  this->l_joint_->SetVelocity(0, this->ljv_);
  this->r_joint_->SetVelocity(0, this->rjv_);
  // tell client that velocities have been update
  geometry_msgs::Point32 current_input;
  current_input.x = this->ljv_;
  current_input.y = this->rjv_;
  this->pub_.publish(current_input);
  //this->timer = this->rosnode_->createTimer(ros::Duration(_msg->z), boost::bind(&GazeboRosForce::ResetForce, this, _1), true);
} 

void GazeboRosForce::ResetForce(const ros::TimerEvent& event) {
  std::cout << "reseting forces.." << std::endl;
  this->l_joint_->SetVelocity(0, 0);
  this->r_joint_->SetVelocity(0, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Update the controller
void GazeboRosForce::UpdateChild()
{
}

// Custom Callback Queue
////////////////////////////////////////////////////////////////////////////////
// custom callback queue thread

void GazeboRosForce::QueueThread()
{
  static const double timeout = 0.05;

  while (this->rosnode_->ok())
  {
    this->queue_.callAvailable(ros::WallDuration(timeout));
  }
}

}


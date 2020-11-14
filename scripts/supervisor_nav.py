#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String, Bool
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from visualization_msgs.msg import Marker
from asl_turtlebot.msg import DetectedObject
import tf
import tf2_ros #do we still need this?
import math
from enum import Enum
from utils import wrapToPi
import numpy as np

# if sim is True/using gazebo, therefore want to subscribe to /gazebo/model_states\
# otherwise, they will use a TF lookup (hw2+)
use_gazebo = rospy.get_param("sim")

# if using gmapping, you will have a map frame. otherwise it will be odom frame
mapping = rospy.get_param("map")

# ---------------------------------------------
#                Constants
# ---------------------------------------------

# threshold at which we consider the robot at a location
POS_EPS = .1
THETA_EPS = .3

# time to stop at a stop sign
STOP_TIME = 3

# minimum distance from a stop sign to obey it
STOP_MIN_DIST = 0.65 #0.5

# time taken to cross an intersection
CROSSING_TIME = 10

# the number of food items
FOOD_ITEMS = 5

# state machine modes, not all implemented
class Mode(Enum):
    # moving
    IDLE = 1
    POSE = 2
    STOP = 3
    CROSS = 4
    NAV = 5   

# food indices
HOT_DOG = 0
APPLE = 1
ORANGE = 2
CAKE = 3
BANANA = 4

print "supervisor settings:\n"
print "use_gazebo = %s\n" % use_gazebo
print "mapping = %s\n" % mapping

class Supervisor:

    def __init__(self):
        rospy.init_node('turtlebot_supervisor', anonymous=True)
        # initialize variables
        self.x = 0
        self.y = 0
        self.theta = 0
        self.goal_update = False
        
        self.mode = Mode.IDLE
        self.last_mode_printed = None
        self.trans_listener = tf.TransformListener()
        #self.trans_broadcaster = tf.TransformBroadcaster()
        #list of the food and it's location
        self.food_data = np.zeros((FOOD_ITEMS, 5))
        self.food_found = [0, 0, 0, 0, 0]
        self.exploring = True
        
        # delivery location
        self.squirtle_x = 3.15 
        self.squirtle_y = 1.6
        self.squirtle_th = 0.0
        
        # home location
        self.home_x = 3.15
        self.home_y = 1.6
        self.home_th = 0.0
        
        self.chunky_radius = 0.1 
        
        # ------------------------
        #       publishers
        # ------------------------
        
        # command pose for controller
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        # nav pose for controller
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        # command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # state machine interface
        self.sm_interface_publisher = rospy.Publisher('/post/nav_fsm', Bool, queue_size =10)
        # for publishing foor markers
        self.vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
        
        # ------------------------
        #       subscribers
        # ------------------------
        # service post topic
        rospy.Subscriber('/post/supervisor_fsm', String, self.post_callback)
        
        # stop sign detector
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        # hot dog detector
        rospy.Subscriber('/detector/hot_dog', DetectedObject, self.hot_dog_detected_callback)
        # apple detector
        rospy.Subscriber('/detector/apple', DetectedObject, self.apple_detected_callback) 
        # orange detector
        rospy.Subscriber('/detector/orange', DetectedObject, self.orange_detected_callback) 
        # cake detector
        rospy.Subscriber('/detector/cake', DetectedObject, self.cake_detected_callback) 
        # banana detector
        rospy.Subscriber('/detector/banana', DetectedObject, self.banana_detected_callback) 
        # turtlebot fsm
        rospy.Subscriber('/post/squirtle_fsm', String, self.post_explore_callback)
        '''
        #[Object]
        rospy.Subscriber('/detector/[object]', DetectedObject, self.[object]_detected_callback)
        #[Object]
        rospy.Subscriber('/detector/[object]', DetectedObject, self.[object]_detected_callback)
        '''
        # high-level navigation pose
        rospy.Subscriber('/nav_pose', Pose2D, self.nav_pose_callback)
        # if using gazebo, we have access to perfect state
        if use_gazebo:
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        # we can subscribe to nav goal click
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        
    # ---------------------------------------------
    #       General Subscriber Callbacks
    # ---------------------------------------------
    # reads the item string and sets up the goal for it.        
    def post_callback(self, msg):
        rospy.loginfo("[SUPERVISOR]: Post rxed: %s", msg.data)
        idx = -1
        isSquirtle = False
        if msg.data == "hotdog":
            idx = 0
        elif msg.data == "apple":
            idx = 1
        elif msg.data == "orange":
            idx = 2
        elif msg.data == "cake":
            idx = 3
        elif msg.data == "banana":
            idx = 4
        elif msg.data == "squirtle":
            pass
        else:
            raise Exception('This item is not supported: %s'
                % msg.data)
        rospy.loginfo("The index is: %d", idx)
        self.goal_update = True 
        if msg.data == "squirtle":
            rospy.loginfo("Setting goal to delivery")
            self.x_g = self.squirtle_x 
            self.y_g = self.squirtle_y
            self.theta_g = self.squirtle_th
        else:
            #populate the goal state based on where we think the food is:
            self.x_g = self.food_data[idx][0]
            self.y_g = self.food_data[idx][1]
            self.theta_g =  self.food_data[idx][2]
        
        #now publish to cmd nav
        self.nav_to_pose()
        self.mode = Mode.NAV
        
    def post_explore_callback(self,msg): 
        rospy.loginfo("[SUPERVISOR]: Received msg: %s", msg.data)
        # check message string
        if msg.data == "done_exploring":
            self.exploring = False
        
    def gazebo_callback(self, msg):
        pose = msg.pose[msg.name.index("turtlebot3_burger")]
        twist = msg.twist[msg.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]
    
    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        origin_frame = "/map" if mapping else "/odom"
        print("rviz command received!")
        try:
            
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            self.x_g = nav_pose_origin.pose.position.x
            self.y_g = nav_pose_origin.pose.position.y
            self.goal_update = True
            quaternion = (
                    nav_pose_origin.pose.orientation.x,
                    nav_pose_origin.pose.orientation.y,
                    nav_pose_origin.pose.orientation.z,
                    nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
        self.mode = Mode.NAV

    def nav_pose_callback(self, msg):
        self.x_g = msg.x
        self.y_g = msg.y
        self.theta_g = msg.theta
        self.mode = Mode.NAV
        
    
    # ---------------------------------------------
    #         Food Detection Callbacks
    # ---------------------------------------------
    
    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance
        rospy.loginfo("Stop Sign found at %f distance", dist)
        # if close enough and in nav mode, stop
        if dist > 0 and dist < STOP_MIN_DIST and self.mode == Mode.NAV:
            self.init_stop_sign()
            
    def hot_dog_detected_callback(self, msg):
    
        #rospy.loginfo("Found hot diggity dog")
        if self.add_food_to_list(msg,HOT_DOG):
            rospy.loginfo("Succesfully added the hot dog")
        #else:
            #rospy.loginfo("Did not add hot dog")
            
    def apple_detected_callback(self, msg):
    
        #rospy.loginfo("Found apple")
        if self.add_food_to_list(msg,APPLE):
            rospy.loginfo("Succesfully added the apple")
        #else:
            #rospy.loginfo("Did not add apple")
            
    def orange_detected_callback(self, msg):
    
        #rospy.loginfo("Found orange")
        if self.add_food_to_list(msg,ORANGE):
            rospy.loginfo("Succesfully added the orange")
        #else:
            #rospy.loginfo("Did not add orange")
            
    def cake_detected_callback(self, msg):
    
        #rospy.loginfo("Found cake")
        if self.add_food_to_list(msg,CAKE):
            rospy.loginfo("Succesfully added the cake")
        #else:
            #rospy.loginfo("Did not add cake")
            
    def banana_detected_callback(self, msg):
    
        #rospy.loginfo("Found banana")
        if self.add_food_to_list(msg,BANANA):
            rospy.loginfo("Succesfully added the banana")
        #else:
            #rospy.loginfo("Did not add banana")
            
    # ---------------------------------------------
    #             Helper Functions
    # ---------------------------------------------
        
    def add_food_to_list(self, msg, label):
        '''
        Description:Add the food item to the data matrix
        Arguments:msg from the topic, food item label
        Returns:False if nothing was added, true if added
        '''
        # get the angle of the frame wrt the world        
        theta_food = 0.5*wrapToPi(msg.thetaleft-msg.thetaright) + self.theta
        
        # find the x, y, of the food using the angle
        x_food = self.x #+(msg.distance - self.chunky_radius)*np.cos(theta_food) 
        y_food = self.y #+(msg.distance - self.chunky_radius)*np.sin(theta_food) 
        # check to see if the food was added or we have a better distance, but only when we are exploring
        if self.exploring and (self.food_found[label] is 0 or msg.distance < self.food_data[label,3]):#np.abs(theta_food) < self.food_data[label,2]:           
            
            # popluate the array at the correct row
            self.food_data[label] = x_food, y_food, theta_food, msg.distance, msg.confidence
            
            # indicate that we found the food
            self.food_found[label] = 1
            
            # add marker to location of food
            #self.broadcast_tf(x_food,y_food,0)
            self.add_marker(x_food, y_food, label)
            
            # return true to indicate successful addition
            return True
            
        # else return false
        else:
            return False
            
    def add_marker(self, x, y, label):
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = label

        marker.type = 2 # sphere

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0 # Don't forget to set the alpha!
        
        if label == HOT_DOG:
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.6
        elif label == APPLE:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif label == ORANGE:
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
        elif label == CAKE:
            marker.color.r = 0.4
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif label == BANANA:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        
        self.vis_pub.publish(marker)
        rospy.loginfo('Published marker!')
            
    def broadcast_tf(self,x,y,th):
        tf2_ros.StaticTransformBroadcaster().sendTransform((x,y,0.0),
                            tf.transformations.quaternion_from_euler(0,0,th),
                            rospy.Time.now(),
                            "/food_marker",
                            "/map")

    def go_to_pose(self):
        """ sends the current desired pose to the pose controller """
        
        pose_g_msg = Pose2D()
        pose_g_msg.x = self.x_g
        pose_g_msg.y = self.y_g
        pose_g_msg.theta = self.theta_g

        self.pose_goal_publisher.publish(pose_g_msg)

    def nav_to_pose(self):
        """ sends the current desired pose to the navigator """
        if self.goal_update:
            self.goal_update = False
            nav_g_msg = Pose2D()
            nav_g_msg.x = self.x_g
            nav_g_msg.y = self.y_g
            nav_g_msg.theta = self.theta_g
            #rospy.loginfo("[SUPERVISOR]: publishing to cmd_nav")
            self.nav_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        """
        vel_g_msg.linear.x = 0.
        vel_g_msg.linear.y = 0.
        vel_g_msg.linear.z = 0.
        vel_g_msg.angular.x = 0.
        vel_g_msg.angular.y = 0.
        vel_g_msg.angular.z = 0.
        """
        
        self.cmd_vel_publisher.publish(vel_g_msg)

    def close_to(self,x,y,theta):
        """ checks if the robot is at a pose within some threshold """

        return (abs(x-self.x)<POS_EPS and abs(y-self.y)<POS_EPS and abs(theta-self.theta)<THETA_EPS)

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """
        # tell nav to idle
        msg = Bool()
        msg.data = True
        self.sm_interface_publisher.publish(msg)
        
        self.goal_update = True
        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return (self.mode == Mode.STOP and (rospy.get_rostime()-self.stop_sign_start)>rospy.Duration.from_sec(STOP_TIME))
        
    def init_crossing(self):
        """ initiates an intersection crossing maneuver """
        # tell nav to resume
        msg = Bool()
        msg.data = False
        self.sm_interface_publisher.publish(msg)
        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return (self.mode == Mode.CROSS and (rospy.get_rostime()-self.cross_start)>rospy.Duration.from_sec(CROSSING_TIME))


    # ---------------------------------------------
    #                State Machine
    # ---------------------------------------------


    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        if not use_gazebo:
            try:
                origin_frame = "/map" if mapping else "/odom"
                (translation,rotation) = self.trans_listener.lookupTransform(origin_frame, '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        # logs the current mode
        if not(self.last_mode_printed == self.mode):
            rospy.loginfo("[SUPERVISOR]: Current Mode: %s", self.mode)
            self.last_mode_printed = self.mode

        # checks wich mode it is in and acts accordingly
        if self.mode == Mode.IDLE:
            # send zero velocity
            self.stay_idle()

        elif self.mode == Mode.POSE:
            # moving towards a desired pose
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.IDLE
            else:
                #self.go_to_pose()
                self.nav_to_pose()

        elif self.mode == Mode.STOP:
            # at a stop sign
            if self.has_stopped():
                self.init_crossing()
            else:
                self.stay_idle()

        elif self.mode == Mode.CROSS:
            # crossing an intersection
            if self.has_crossed():
                self.mode = Mode.POSE
            else:
                self.nav_to_pose()

        elif self.mode == Mode.NAV:
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.IDLE
            else:
                self.nav_to_pose()

        else:
            raise Exception('This mode is not supported: %s'
                % str(self.mode))

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

if __name__ == '__main__':
    sup = Supervisor()
    sup.run()
    

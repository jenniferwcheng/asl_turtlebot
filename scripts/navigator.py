#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj
from grids import StochOccupancyGrid2D
import scipy.interpolate
import scipy.ndimage.morphology as morpho
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# size of buffer
CMD_HISTORY_SIZE = 25

INFLATE_TIME = 5

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    BACKING_UP = 4
    INFLATE = 5
    INFLATE_ALIGN = 6

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)
        self.mode = Mode.IDLE
        self.mode_at_stop = None
        self.x_saved = None
        self.y_saved = None
        self.theta_saved = None

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # history tracking of controls
        self.history_cnt = 0
        self.V_history =  np.zeros(CMD_HISTORY_SIZE)
        self.om_history = np.zeros(CMD_HISTORY_SIZE)
        self.backing_cnt = 0
        
        
        #laser scans for collision
        self.laser_ranges = []
        self.laser_angle_increment = 0.01 # this gets updated
        self.chunky_radius = 0.11 #TODO: Tune this! FYI 0.12 sucks so bad -KJ
        
        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0
        
        self.iters = 0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.map_probs_inflated = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution =  0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.,0.]
        
        # Robot limits
        self.v_max = rospy.get_param("~v_max", 0.2)  #0.2  # maximum velocity
        self.om_max =rospy.get_param("~om_max", 0.4)  #0.4   # maximum angular velocity

        self.v_des = 0.12   # desired cruising velocity
        self.theta_start_thresh = 0.05   # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.2     # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = (2.0*np.pi)/20.0 #0.05

        # trajectory smoothing
        self.spline_alpha = 0.01
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.

        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)
        self.pose_controller = PoseController(0., 0., 0., self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)
        
        # timing variables
        self.start_time = 0.0
        self.wait_time = None
        
        # Publishers
        self.nav_planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher('/cmd_smoothed_path', Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher('/cmd_smoothed_path_rejected', Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)
        #communication with squirtle_fsm to inform goal status
        self.publish_squirtle = rospy.Publisher('/post/squirtle_fsm', String, queue_size = 10)
         
        # Subscriber Constructors
        rospy.Subscriber('/post/nav_fsm', Bool, self.post_callback) #service queue
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback) 
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('debug/nav_fsm', String, self.debug_callback)

        print "finished init"
     
    #------------------------------------------------------------------
    # Subscriber Callbacks
    #------------------------------------------------------------------
    #for in terminal debug
    def debug_callback(self,msg):
        if msg.data == "query_goal":
            print('[NAV DEBUG]: The goal is: %f, %f, %f' %(self.x_g, self.y_g, self.theta_g))
        elif msg.data == "query_state":
            print("[NAV DEBUG]: The current state is: %s" %str(self.mode))
        elif msg.data == "why_u_do_dis?":
            print("[NAV DEBUG]: The current position delta is: %f, %f, %f" %(self.x_g - self.x, self.y_g - self.y, self.theta_g-self.theta))
        else:
            print("[NAV DEBUG]: Invalid debug message")
            
    #for the interface topic between nav and supervisor 
    def post_callback(self,data):
        rospy.loginfo("Received from interface topic")
        
        # received true = stop
        if data.data is True:   
                        
            # store current goal
            self.x_saved = self.x_g
            self.y_saved = self.y_g
            self.theta_saved = self.theta_g
            
            # set goal to nothing
            self.x_g = None
            self.y_g = None
            self.theta_g = None
            
            #put machine back into idle
            self.switch_mode(Mode.IDLE)
      
        return
        
    # for getting the laser data    
    def laser_callback(self, msg):
        """ callback for thr laser rangefinder """
        
        self.laser_ranges = msg.ranges
        self.laser_angle_increment = msg.angle_increment
            
    def dyn_cfg_callback(self, config, level):
        rospy.loginfo("Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}, spline_alpha:{spline_alpha}".format(**config))
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        self.spline_alpha = config["spline_alpha"]
        
        self.traj_controller.kpx = config["kpx"]
        self.traj_controller.kpy = config["kpy"]
        self.traj_controller.kdx = config["kdx"]
        self.traj_controller.kdy = config["kdy"]
        self.traj_controller.V_max = config["V_max"]
        self.traj_controller.om_max = config["om_max"]
        
        self.chunky_radius = config["chunky_radius"]
        #rospy.loginfo("[DYN CONFIG] Chunky_radius: %f",self.chunky_radius)
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if data.x != self.x_g or data.y != self.y_g or data.theta != self.theta_g:
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()
            
            #tell squirtle we are no longer at the goal
            #msg = String()
            #msg.data = "not_at_goal"
            #self.publish_squirtle.publish(msg)

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs)
            #binary_map = (self.map_probs > 90)
            #mask = scipy.binary_dilation(binary_map)
            #inflated_map = 100*(mask ^ (self.map_probs > 90)) + self.map_probs
            
            #print(self.map_width)
            #print(self.map_height)
            #print(self.map_resolution)
          
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                if (self.mode is not Mode.BACKING_UP) and (self.mode is not Mode.INFLATE) and (self.mode is not Mode.INFLATE_ALIGN):
                    rospy.loginfo("replanning because of new map")
                    self.replan() # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        # if the goal is none, it's already there you dumb Winnebago
        if self.x_g is None:
            return True
        return (linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.at_thresh and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta)

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh)
        
    def close_to_plan_start(self):
        return (abs(self.x - self.plan_start[0]) < self.start_pos_thresh and abs(self.y - self.plan_start[1]) < self.start_pos_thresh)

    def snap_to_grid(self, x):
        return (self.plan_resolution*round(x[0]/self.plan_resolution), self.plan_resolution*round(x[1]/self.plan_resolution))

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode
    #------------------------------------------------------------------
    # Publishers
    #------------------------------------------------------------------
    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i,0]
            pose_st.pose.position.y = traj[i,1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        #-------------------------------------
        #  helper functions for LIFO queue
        #-------------------------------------
        
        def enQ_buffer(V_in, om_in):
        
            #rospy.loginfo("Enqueuing controls")
            # roll right 1 [n]->[n+1]
            self.V_history = np.roll(self.V_history, 1, axis=None)
            self.om_history = np.roll(self.om_history, 1, axis=None)
            # load in the value
            self.V_history[0] = V_in
            self.om_history[0] = om_in
                    
        def deQ_buffer():
            #rospy.loginfo("Dequeuing controls")
            # get first value out
            V_out = -1.0*self.V_history[0] 
            om_out = -1.0*self.om_history[0]
            # roll left 1 [n]<-[n+1]
            self.V_history = np.roll(self.V_history, -1, axis=None)
            self.om_history = np.roll(self.om_history, -1, axis=None)
            return V_out, om_out
            
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.TRACK or self.mode == Mode.INFLATE:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.ALIGN or self.mode == Mode.INFLATE_ALIGN:
            V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.BACKING_UP and self.backing_cnt < (CMD_HISTORY_SIZE-1):
            #extract elements from the queue
            V, om = deQ_buffer()
        else:
            V = 0.
            om = 0.
        
        #enqueue for history tracking
        if self.mode is not Mode.BACKING_UP:
            enQ_buffer(V, om)
            
            
        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0
        
    def inflate_map(self):
        rospy.loginfo("[NAVIGATOR]: Inflating Map")
        
        # convert to matrix
        map_matrix = np.reshape(self.map_probs, (384,384)) #TODO: Magic numbers
        map_matrix_inflated = morpho.grey_dilation(map_matrix, size=(3,3))
        
        # flatten back to list
        self.map_probs_inflated = map_matrix_inflated.flatten().tolist()
        
        if self.map_width>0 and self.map_height>0 and len(self.map_probs_inflated)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs_inflated)
    
    def deflate_map(self):
        rospy.loginfo("[NAVIGATOR]: Deflating Map")
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs)
    
    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("Navigator: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

        rospy.loginfo("Navigator: computing navigation plan")
        
        success =  problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            #tell squirtle we could not find a path
            msg = String()
            msg.data = "no_path"
            self.publish_squirtle.publish(msg)
            #TODO: THIS WAS COMMENTED OUT BEFORE, BUT WAS HERE IN RECENT COMMIT
            #ATTEMPTING TO UNCOMMENT AGAIN AND TEST!
            if self.mode == Mode.BACKING_UP:
                self.deflate_map()
                self.switch_mode(Mode.IDLE)
            return
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path

        # Check whether path is too short
        if self.at_goal():
            rospy.loginfo("Path already at goal pose")
            #tell squirtle we are at the goal
            msg = String()
            msg.data = "at_goal"
            self.publish_squirtle.publish(msg)
            self.switch_mode(Mode.IDLE)
            return
        elif len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = self.current_plan_duration - self.get_current_plan_time()

            # Estimate duration of new trajectory
            th_init_new = traj_new[0,2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err/self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo("New plan rejected (longer duration than current plan)")
                self.publish_smoothed_path(traj_new, self.nav_smoothed_path_rej_pub)
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0,2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            if self.mode == Mode.BACKING_UP:
                self.switch_mode(Mode.INFLATE_ALIGN)
            else:
                self.switch_mode(Mode.ALIGN)
            return
            
        if self.mode == Mode.BACKING_UP: #what happens if we want to nav to new goal here...?
            self.start_timer(INFLATE_TIME)
            self.switch_mode(Mode.INFLATE)
        else:
            rospy.loginfo("Ready to track")
            self.switch_mode(Mode.TRACK)
    
    def start_timer(self,duration):
    
        # set the duration
        self.wait_time = duration
        # get sys time at start
        self.start_time = rospy.get_rostime()  
    
    def is_time_expired(self):
        returnVal = False
        # check the timer is running
        if self.wait_time is not None:
            # see if the timer is expired
            if (rospy.get_rostime()-self.start_time) > rospy.Duration.from_sec(self.wait_time):
                self.wait_time = None
                returnVal = True
        
        return returnVal
        
    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print e
                pass
            
            """  
            def if_about_to_hit_wall():
                
                # check if inflated turtlebot circumference will hit the wall
                #th_offset = np.pi/8.0
                #arr = np.arange(wrapToPi(self.theta-th_offset), wrapToPi(self.theta+th_offset), 0.02)
                arr = np.arange(0, 2*np.pi, 0.02)
                for i in range(len(arr)):
                    if not self.occupancy.is_free(np.array([self.x + np.cos(arr[i])*self.chunky_radius,self.y+np.sin(arr[i])*self.chunky_radius])):
                        return True
                return False  
                
                #return not self.occupancy.is_free(np.array([self.x ,self.y]))
            """ 
            def if_about_to_hit_wall(laserRanges):
                #initialize return value to false
                returnFlag = False
                ##remove the zeros
                #validRanges = np.trim_zeros(laserRanges)
                #validRanges = [i for i, dist in enumerate(laserRanges) if dist != 0]
                #check the minimum scan distance
                #rospy.loginfo(laserRanges)
                minScanDist = min(laserRanges)
                #rospy.loginfo("Minimum Scan Distance: %f, radius: %f", minScanDist, self.chunky_radius)
                
                #see if less that our boy's fat body
                if minScanDist < self.chunky_radius:
                    #set flag to true
                    returnFlag = True 
                #return the flag                 
                return returnFlag
            
            def purge_control_queue():
                # NUKE EVERYTHING!!!!!!!!!
                self.history_cnt = 0
                self.V_history =  np.zeros(CMD_HISTORY_SIZE)
                self.om_history = np.zeros(CMD_HISTORY_SIZE)
                self.backing_cnt = 0

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
                    
            elif self.mode == Mode.INFLATE_ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.start_timer(INFLATE_TIME)
                    self.switch_mode(Mode.INFLATE)
                    
            elif self.mode == Mode.TRACK:
                if if_about_to_hit_wall(self.laser_ranges):
                    rospy.loginfo("About to hit a wall")
                    self.start_timer(10)
                    self.switch_mode(Mode.BACKING_UP)
                elif self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan() # we aren't near the goal but we thought we should have been, so replan
                    
            elif self.mode == Mode.PARK:
                #if if_about_to_hit_wall():
                 #   self.switch_mode(Mode.BACKING_UP)
                if self.at_goal():
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)
                    
                    purge_control_queue()
                    
                    #tell squirtle we are at the goal
                    msg = String()
                    msg.data = "at_goal"
                    self.publish_squirtle.publish(msg)
            
            elif self.mode == Mode.BACKING_UP:
                # see if we have backed enough counts
                if (self.backing_cnt >= CMD_HISTORY_SIZE):# or if_about_to_hit_wall(self.laser_ranges):
                    self.backing_cnt = CMD_HISTORY_SIZE #just for good measure
                    #stop the robot
                    self.publish_control()
                    #purge the control since backup is done 
                    purge_control_queue()
                    
                    #make walls chunkier
                    self.inflate_map()
                    
                    self.replan() #switches mode internally
                    #self.switch_mode(Mode.TRACK)
                    
                elif self.is_time_expired():
                    purge_control_queue()

                    
                    #we must have gotten stuck or something
                    #we need to run replan again 
                    self.deflate_map()
                    self.switch_mode(Mode.TRACK)
                    self.replan()
                    
                    
                else:
                    # increment count
                    self.backing_cnt += 1
            elif self.mode == Mode.INFLATE:
                if self.is_time_expired():
                    rospy.loginfo("Inflate time expired")
                    self.deflate_map()
                    self.replan()
                elif self.near_goal():
                    self.deflate_map()
                    self.switch_mode(Mode.PARK)    
            
            self.publish_control()
            rate.sleep()

if __name__ == '__main__':    
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)

    nav.run()

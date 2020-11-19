#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String, Bool
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from visualization_msgs.msg import Marker
from asl_turtlebot.msg import DetectedObject
import tf
import math
from enum import Enum
from utils import wrapToPi
import numpy as np
import time

# state machine modes
class Mode(Enum):
    # food delivery
    EXPLORE = 0
    WAIT_FOR_ORDER = 1 
    NAV_2_PICKUP = 2
    PICKING_UP = 3
    NAV_2_DELIV = 4 
    DELIVERING = 5
    NAV_2_WAYPT = 6
     
WAIT_TIME = 5 # [s]   
DELIVER_TIME = 5 # [s]

class Squirtle:

    def __init__(self):
        rospy.init_node('squirtleFSM', anonymous=True)
        # ------------------------
        #       variables
        # ------------------------
    
        # state machine variables
        self.mode = Mode.EXPLORE
        self.last_mode_printed = None
        
        # flags
        self.is_at_goal = False
        
        # delivery variables
        self.order_items = [] #order list from user
        self.item_queue = [] #queue for items where a path could not be planned
        self.current_item = None
        self.num_items = 0
        self.current_order = 0
        
        # timing variables
        self.start_time = 0.0
        self.wait_time = None
        
        # ------------------------
        #       publishers
        # ------------------------
        
        # state machine interface
        self.supervisor_fsm_pub = rospy.Publisher('/post/supervisor_fsm', String, queue_size =10)
        # debugging
        self.debug_pub = rospy.Publisher('/debug/squirtle_fsm',String,queue_size=10)
        
        # ------------------------
        #       subscribers
        # ------------------------
        rospy.Subscriber('/post/squirtle_fsm', String, self.post_callback)
        rospy.Subscriber('/debug/squirtle_fsm', String, self.debug_callback)
        rospy.Subscriber('/delivery_request', String, self.delivery_callback)
    
    # ---------------------------
    #    Subscriber callbacks
    # ---------------------------
    
    def post_callback(self,msg): 
        rospy.loginfo("[SQUIRTLE]: Received msg: %s", msg.data)
        # check message string
        if msg.data == "at_goal":
            self.is_at_goal = True
        elif msg.data == "not_at_goal":
            self.is_at_goal = False
        elif msg.data == "done_exploring":
            rospy.loginfo("Handling: done_exploring")
            self.switch_mode(Mode.WAIT_FOR_ORDER)
        #if user gives squirtle a new waypoint outside of EXPLORE mode
        elif msg.data == "way_point" and self.mode is not Mode.EXPLORE:
            #switch state to NAV_2_WAYPT
            self.switch_mode(Mode.NAV_2_WAYPT)
        elif msg.data == "no_path" and self.mode == Mode.NAV_2_PICKUP:
            rospy.loginfo("Current item not obtainable")
            self.item_queue.append(self.current_item)
            self.debug_pub.publish("query_queue") #DEBUGGING QUEUE
            if self.pickup_order():
                self.check_item_queue()
            
    def debug_callback(self,msg):
        if msg.data == "query_state":
            print("[SQUIRTLE DEBUG]: Mode:%s" %str(self.mode))
        elif msg.data == "query_queue":
            print("[SQUIRTLE DEBUG]: Queue: %s" %str(self.item_queue))
        
    def delivery_callback(self,msg):
        # only accept deliveries if we are in WAIT_FOR_ORDER mode
        if self.mode == Mode.WAIT_FOR_ORDER:
            # make sure order is not empty
            if msg.data == '':
                rospy.loginfo("No order received")
                #return
            else:
                # split order into separate strings, comma delimited ["a,b,c"] -> ["a","b","c"]    
                self.order_items = msg.data.strip().split(',')
                rospy.loginfo("List: %s", str(self.order_items))
                self.num_items = len(self.order_items)
                rospy.loginfo("Number of items to pickup: %s", str(self.num_items))
                
                #go get the first item
                self.pickup_order()
                   
        
    # ------------------------
    #    Helper Functions
    # ------------------------  
     
    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode
    
    def send_item(self):
        # clear the goal status flag
        self.is_at_goal = False
        # tell supervisor which order to pick up
        self.supervisor_fsm_pub.publish(self.current_item)
        #self.supervisor_fsm_pub.publish(self.order_items[self.num_items].strip())
        # switch to picking up state
        self.switch_mode(Mode.NAV_2_PICKUP)
    
    def pickup_order(self):
        #check to see if we still have items in our list
        if not self.order_items:
            rospy.loginfo("All items picked up")
            #return true to indicate list is complete
            return True
        else: # if we have items to pickup
            rospy.loginfo("More items to get")
            #pop off the list
            self.current_item = self.order_items.pop(0) #FIFO
            #send item to supervisor
            self.send_item()
            # return false to indicate list not empty
            return False
    '''        
    def pickup_order(self):
        """
        Argument: self
        Return: true if all orders have been picked up
        
        Looks at the item list, grabs the top item for pickup
        """
        if self.num_items is not 0:
            rospy.loginfo("More items to get")
            # decrement list count
            self.num_items -= 1
            self.current_item = self.order_items[self.num_items].strip()
            self.send_item()
            """
            # clear the goal status flag
            self.is_at_goal = False
            # tell supervisor which order to pick up
            self.supervisor_fsm_pub.publish(self.current_item)
            #self.supervisor_fsm_pub.publish(self.order_items[self.num_items].strip())
            # switch to picking up state
            self.switch_mode(Mode.NAV_2_PICKUP)
            """
            # return false to indicate list not empty
            return False
        else:
            rospy.loginfo("All items picked up")
            #return true to indicate list is complete
            return True
    '''
    def check_item_queue(self):
        """
        Arguments: self
        Returns true if there was an item that was in the queue that needed to be picked up
        """
        #check if queue is empty
        if not self.item_queue:
            #indicate empty queue to terminal
            print("[SQUIRTLE], queue is empty!")
            return False
        #else we have items in the queue we need to plan to
        else:
            rospy.loginfo("[SQUIRTLE], popping from queue")
            #extract first element in the queue
            self.current_item = self.item_queue.pop(0)
            self.debug_pub.publish("query_queue") #DEBUGGING QUEUE
            #send item to supervisor for pickup
            self.send_item()
            #set return value to true to indicate we popped something from the queue
            return True
                  
            
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
                
    def deliver_order(self):
        # tell supervisor to go to squirtle
        self.supervisor_fsm_pub.publish("squirtle")
        # clear the goal status flag
        self.is_at_goal = False
        # switch to navigating to delivery location
        self.switch_mode(Mode.NAV_2_DELIV)
        
    def is_done_exploring(self):
        return self.mode == Mode.EXPLORE
    
        
    # ----------------------------
    #    Machine Run Function
    # ----------------------------
    def loop(self):

        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        # logs the current mode
        if not(self.last_mode_printed == self.mode):
            rospy.loginfo("[SQUIRTLE]: Current Mode: %s", self.mode)
            self.last_mode_printed = self.mode

        # checks wich mode it is in and acts accordingly
        if self.mode == Mode.EXPLORE:
            # This is handled in the post_callback
            pass
             
        elif self.mode == Mode.WAIT_FOR_ORDER:
            # This is handled in delivery_callback
            pass
            
        elif self.mode == Mode.NAV_2_PICKUP:
            # check if at goal
            if self.is_at_goal:
                # start pickup timer
                self.start_timer(WAIT_TIME)
                # switch mode
                self.switch_mode(Mode.PICKING_UP)  
                    
        elif self.mode == Mode.PICKING_UP:
            # check to see if the pickup time is over
            if self.is_time_expired():
                # finished picking up, so reset goal
                self.is_at_goal = False
                #check item queue
                if self.check_item_queue():
                    pass
                elif self.pickup_order(): # returns true if we finished picking up all the orders
                    # we are done, so switch to delivery
                    self.deliver_order() # transition occurs in this function
                    #purge to order queue
                    #self.item_queue.clear()  
                          
        elif self.mode == Mode.NAV_2_DELIV:
            if self.is_at_goal:
                #start pickup timer
                self.start_timer(DELIVER_TIME)
                #switch mode
                self.switch_mode(Mode.DELIVERING)
            
        elif self.mode == Mode.DELIVERING:
            # check to see if the delivery time is over
            if self.is_time_expired():
                # finished delivering up, so reset goal
                self.is_at_goal = False
                # finished with delivery so we can go back to waiting for order
                self.switch_mode(Mode.WAIT_FOR_ORDER)
        elif self.mode == Mode.NAV_2_WAYPT:
            if self.is_at_goal:
                #resume back to current item for pickup
                self.send_item()
        else:
            raise Exception('This mode is not supported: %s'
                % str(self.mode))    

     

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()


if __name__ == '__main__':
    squirtle = Squirtle()
    squirtle.run()

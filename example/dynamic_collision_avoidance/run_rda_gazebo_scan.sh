#!/bin/bash

# Step 1: Launch the Gazebo environment
gnome-terminal -- bash -c "roslaunch rda_ros gazebo_limo_env20.launch"

# Give some time for Gazebo to start
sleep 15

# Step 2: Launch the RDA control with obstacle information from laser scan
gnome-terminal -- bash -c "roslaunch rda_ros rda_gazebo_limo_scan.launch"
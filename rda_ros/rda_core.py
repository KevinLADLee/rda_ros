#! /usr/bin/env python

import cv2
import numpy as np
from collections import namedtuple
from math import atan2, cos, sin
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
import rclpy.publisher
import rclpy.time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from costmap_converter_msgs.msg import ObstacleArrayMsg

from RDA_planner.mpc import MPC
from gctl.curve_generator import curve_generator

robot_tuple = namedtuple(
    "robot_tuple", "G h cone_type wheelbase max_speed max_acce dynamics"
)
rda_obs_tuple = namedtuple(
    "rda_obs_tuple", "center radius vertex cone_type velocity"
)  # vertex: 2*number of vertex


class rda_core(Node):
    def __init__(self) -> None:
        super().__init__('rda_node')

        # publish topics
        self.vel_pub = self.create_publisher(Twist, '/rda_cmd_vel', 10)
        self.rda_path_pub = self.create_publisher(Path, '/rda_opt_path', 10)
        self.ref_path_pub = self.create_publisher(Path, '/rda_ref_path', 10)
        self.ref_states_pub = self.create_publisher(Path, '/rda_ref_states', 10)
        self.obs_pub = self.create_publisher(MarkerArray, '/rda_obs_markers', 10)

        ## robot info
        self.declare_parameters(
            namespace='',
            parameters=[
            ("robot_info.vertices", []),
            ("robot_info.radius", []),
            ("robot_info.max_speed", [10.0, 1.0]),
            ("robot_info.max_acce", [10.0, 0.5]),
            ("robot_info.length", 2),
            ("robot_info.width", 1),
            ("robot_info.wheelbase", 1.5),
            ("robot_info.dynamics", "diff"),
            ("robot_info.cone_type", "Rpositive"),
            ]
        )

        robot_info = {
            "vertices": self.get_parameter("robot_info.vertices").get_parameter_value().double_array_value,
            "radius": self.get_parameter("robot_info.radius").get_parameter_value().double_value,
            "max_speed": self.get_parameter("robot_info.max_speed").get_parameter_value().double_array_value,
            "max_acce": self.get_parameter("robot_info.max_acce").get_parameter_value().double_array_value,
            "length": self.get_parameter("robot_info.length").get_parameter_value().double_value,
            "width": self.get_parameter("robot_info.width").get_parameter_value().double_value,
            "wheelbase": self.get_parameter("robot_info.wheelbase").get_parameter_value().double_value,
            "dynamics": self.get_parameter("robot_info.dynamics").get_parameter_value().string_value,
            "cone_type": self.get_parameter("robot_info.cone_type").get_parameter_value().string_value,
        }

        ## For rda MPC
        receding = self.get_parameter_or("receding", 10)
        iter_num = self.get_parameter_or("iter_num", 2)
        enable_reverse = self.get_parameter_or("enable_reverse", False).get_parameter_value().bool_value
        sample_time = self.get_parameter_or("sample_time", 0.1).get_parameter_value().double_value
        process_num = self.get_parameter_or("process_num", 4).get_parameter_value().integer_value
        accelerated = self.get_parameter_or("accelerated", True).get_parameter_value().bool_value
        time_print = self.get_parameter_or("time_print", False).get_parameter_value().bool_value
        obstacle_order = self.get_parameter_or("obstacle_order", True).get_parameter_value().bool_value
        self.max_edge_num = self.get_parameter_or("max_edge_num", 5).get_parameter_value().integer_value
        self.max_obstacle_num = self.get_parameter_or("max_obs_num", 5).get_parameter_value().integer_value
        self.goal_index_threshold = self.get_parameter_or("goal_index_threshold", 1).get_parameter_value().integer_value

        ## Tune parameters
        iter_threshold = self.get_parameter_or("iter_threshold", 0.2).get_parameter_value().double_value
        slack_gain = self.get_parameter_or("slack_gain", 8).get_parameter_value().integer_value
        max_sd = self.get_parameter_or("max_sd", 1.0).get_parameter_value().double_value
        min_sd = self.get_parameter_or("min_sd", 0.1).get_parameter_value().double_value
        ws = self.get_parameter_or("ws", 1.0).get_parameter_value().double_value
        wu = self.get_parameter_or("wu", 0.5).get_parameter_value().double_value
        ro1 = self.get_parameter_or("ro1", 200).get_parameter_value().integer_value
        ro2 = self.get_parameter_or("ro2", 1.0).get_parameter_value().double_value

        # reference speed
        self.ref_speed = self.get_parameter_or("ref_speed", 4.0).get_parameter_value().double_value  # ref speed

        ## for scan
        use_scan_obstacle = self.get_parameter_or("use_scan_obstacle", False).get_parameter_value().bool_value
        self.scan_eps = self.get_parameter_or("scan_eps", 0.2).get_parameter_value().double_value
        self.scan_min_samples = self.get_parameter_or("scan_min_samples", 6).get_parameter_value().integer_value

        ## for reference paths
        self.waypoints = self.get_parameter_or("waypoints", []).get_parameter_value().double_array_value
        self.loop = self.get_parameter_or("loop", False).get_parameter_value().bool_value
        self.curve_type = self.get_parameter_or("curve_type", "dubins").get_parameter_value().string_value
        self.step_size = self.get_parameter_or("step_size", 0.1).get_parameter_value().double_value
        self.min_radius = self.get_parameter_or("min_radius", 1.0).get_parameter_value().double_value

        ## for frame
        self.target_frame = self.get_parameter_or("target_frame", "map").get_parameter_value().string_value
        self.lidar_frame = self.get_parameter_or("lidar_frame", "lidar_link").get_parameter_value().string_value
        self.base_frame = self.get_parameter_or("base_frame", "base_link").get_parameter_value().string_value

        # for visualization
        self.marker_x = self.get_parameter_or("marker_x", 0.05).get_parameter_value().double_value
        self.marker_lifetime = self.get_parameter_or("marker_lifetime", 0.1).get_parameter_value().double_value

        # init tf
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)

        # initialize
        self.robot_state = None
        self.obstacle_list = []
        self.cg = curve_generator()
        self.ref_path_list = (
            self.generate_ref_path_list()
        )  # generate the initial reference path
        robot_info_tuple = self.generate_robot_tuple(robot_info)

        self.rda_opt = MPC(
            robot_info_tuple,
            self.ref_path_list,
            receding,
            sample_time,
            iter_num,
            enable_reverse,
            False,
            obstacle_order,
            self.max_edge_num,
            self.max_obstacle_num,
            process_num,
            accelerated,
            time_print,
            self.goal_index_threshold,
            iter_threshold=iter_threshold,
            slack_gain=slack_gain,
            max_sd=max_sd,
            min_sd=min_sd,
            ws=ws,
            wu=wu,
            ro1=ro1,
            ro2=ro2,
        )

        self.create_subscription(PoseStamped, "/rda_goal", self.goal_callback, 10)
        self.create_subscription(Path, "/rda_sub_path", self.path_callback, 10)

        # Topic Subscribe
        if not use_scan_obstacle:
            self.create_subscription(ObstacleArrayMsg, "/rda_obstacles", self.obstacle_callback, 10)
        else:
            self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

    def control(self):
        rate = self.create_rate(50)

        while rclpy.ok():

            self.read_robot_state()

            if self.robot_state is None:
                self.get_logger().info("waiting for robot states", throttle_duration_sec=1)
                continue

            if len(self.obstacle_list) == 0:
                self.get_logger().info("No obstacles, perform path tracking", throttle_duration_sec=1)
            else:
                rda_obs_markers = self.convert_to_markers(self.obstacle_list)
                self.obs_pub.publish(rda_obs_markers)

            if self.rda_opt.no_ref_path():
                self.get_logger().info("waiting for reference path, topic '/rda_sub_path'", throttle_duration_sec=1)
                continue

            else:
                ref_path = self.convert_to_path(self.ref_path_list)
                self.ref_path_pub.publish(ref_path)

            if self.max_obstacle_num == 0:
                opt_vel, info = self.rda_opt.control(
                    self.robot_state, self.ref_speed, []
                )
            else:
                opt_vel, info = self.rda_opt.control(
                    self.robot_state, self.ref_speed, self.obstacle_list
                )

            if info["arrive"]:
                if self.loop:

                    self.goal = self.rda_opt.ref_path[0]
                    start = self.rda_opt.ref_path[-1]

                    self.ref_path_list = self.cg.generate_curve(
                        self.curve_type,
                        [start, self.goal],
                        self.step_size,
                        self.min_radius,
                    )
                    print("start new loop")
                    self.rda_opt.update_ref_path(self.ref_path_list)

                else:
                    opt_vel = np.zeros((2, 1))
                    print("arrive at the goal!")

            vel = self.convert_to_twist(opt_vel)
            rda_opt_path = self.convert_to_path(info["opt_state_list"])
            ref_states = self.convert_to_path(info["ref_traj_list"])

            self.ref_states_pub.publish(ref_states)
            self.vel_pub.publish(vel)
            self.rda_path_pub.publish(rda_opt_path)

            rate.sleep()

    def read_robot_state(self):

        try:
            trans = self.tf_buffer.lookup_transform(
            self.target_frame, self.base_frame, rclpy.time.Time()
            )
            yaw = self.quat_to_yaw(trans.transform.rotation)
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            self.robot_state = np.array([x, y, yaw]).reshape(3, 1)

        except TransformException as ex:
            self.get_logger().info(
            f"Could not transform {self.base_frame} to {self.target_frame}: {ex}",
            throttle_duration_sec=1,
            )

    def obstacle_callback(self, obstacle_array):

        temp_obs_list = []

        if self.max_obstacle_num == 0:
            self.get_logger().info("No obstacles are considered")
            return

        for obstacles in obstacle_array.obstacles:

            vertex = obstacles.polygon.points
            vertex_num = len(vertex)

            if vertex_num == 1:
                # circle obstacle

                center = np.array([[vertex[0].x], [vertex[0].y]])
                radius = obstacles.radius

                linear_x, linear_y = (
                    obstacles.velocities.twist.linear.x,
                    obstacles.velocities.twist.linear.y,
                )
                velocity = np.array([[linear_x], [linear_y]])

                circle_obs = rda_obs_tuple(center, radius, None, "norm2", velocity)

                temp_obs_list.append(circle_obs)

            elif vertex_num == 2:
                # line obstacle
                continue

            elif vertex_num > 2:
                # polygon obstacle
                vertex_list = [np.array([[p.x], [p.y]]) for p in vertex]
                vertexes = np.hstack(vertex_list)

                linear_x, linear_y = (
                    obstacles.velocities.twist.linear.x,
                    obstacles.velocities.twist.linear.y,
                )
                velocity = np.array([[linear_x], [linear_y]])

                polygon_obs = rda_obs_tuple(None, None, vertexes, "Rpositive", velocity)

                temp_obs_list.append(polygon_obs)

        self.obstacle_list[:] = temp_obs_list[:]

    def path_callback(self, path):

        self.ref_path_list = []

        for p in path.poses:
            x = p.pose.position.x
            y = p.pose.position.y
            theta = self.quat_to_yaw(p.pose.orientation)

            points = np.array([x, y, theta]).reshape(3, 1)
            self.ref_path_list.append(points)

        if len(self.ref_path_list) == 0:
            self.get_logger().info(
                "No waypoints are converted to reference path, waiting for new waypoints",
                throttle_duration_sec=1,
            )
            return

        self.get_logger().info("reference path update", throttle_duration_sec=0.1)
        self.rda_opt.update_ref_path(self.ref_path_list)

    def goal_callback(self, goal):

        x = goal.pose.position.x
        y = goal.pose.position.y
        theta = self.quat_to_yaw(goal.pose.orientation)

        self.goal = np.array([[x], [y], [theta]])

        print(f"set rda goal: {self.goal}")

        self.ref_path_list = self.cg.generate_curve(
            self.curve_type,
            [self.robot_state, self.goal],
            self.step_size,
            self.min_radius,
        )

        if len(self.ref_path_list) == 0:
            self.get_logger().info(
            "No waypoints are converted to reference path, waiting for new waypoints",
            throttle_duration_sec=1,
            )
            return

        self.get_logger().info("reference path update", throttle_duration_sec=0.1)
        self.rda_opt.update_ref_path(self.ref_path_list)

    def scan_callback(self, scan_data):

        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))

        point_list = []

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan_data.range_max - 0.01):
                point = np.array(
                    [[scan_range * np.cos(angle)], [scan_range * np.sin(angle)]]
                )
                point_list.append(point)

        if len(point_list) < 3 or self.robot_state is None:
            self.get_logger().info("No obstacles are converted to polygon", throttle_duration_sec=1)
            return

        else:

            # get the transform from lidar to target frame
            try:
                trans, rot = self.listener.lookupTransform(
                    self.target_frame, self.lidar_frame, rclpy.time.Time()
                )

                yaw = self.quat_to_yaw_list(rot)
                x, y = trans[0], trans[1]

                l_trans, l_R = self.get_transform(np.array([x, y, yaw]).reshape(3, 1))

            except (
                TransformException
            ):
                self.get_logger().info(
                    f"waiting for tf for the transform from {self.lidar_frame} to {self.target_frame}",
                    throttle_duration_sec=1,
                )
                return

            # convert the points to convex polygon
            point_array = np.hstack(point_list).T
            labels = DBSCAN(
                eps=self.scan_eps, min_samples=self.scan_min_samples
            ).fit_predict(point_array)

            self.obstacle_list = []

            for label in np.unique(labels):
                if label == -1:
                    continue
                else:
                    point_array2 = point_array[labels == label]
                    rect = cv2.minAreaRect(point_array2.astype(np.float32))
                    box = cv2.boxPoints(rect)

                    vertices = box.T

                    global_vertices = l_R @ vertices + l_trans

                    self.obstacle_list.append(
                        rda_obs_tuple(None, None, global_vertices, "Rpositive", 0)
                    )

    def generate_ref_path_list(self):

        if len(self.waypoints) == 0:
            return []

        else:
            point_list = [
                np.array([[p[0]], [p[1]], [p[2]]]).astype("float64")
                for p in self.waypoints
            ]
            ref_path_list = self.cg.generate_curve(
                self.curve_type, point_list, self.step_size, self.min_radius
            )

            return ref_path_list

    def convert_to_markers(self, obs_list):

        marker_array = MarkerArray()

        # obs: center, radius, vertex, cone_type, velocity

        for obs_index, obs in enumerate(obs_list):
            marker = Marker()
            marker.header.frame_id = self.target_frame

            marker.header.stamp = self.get_clock().now()

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0

            marker.lifetime = rclpy.time.Duration(self.marker_lifetime)

            # breakpoint()

            if obs.vertex is not None:

                marker.type = marker.LINE_LIST

                marker.scale.x = self.marker_x

                temp_matrix = np.hstack((obs.vertex, obs.vertex[:, 0:1]))
                for i in range(temp_matrix.shape[1] - 1):
                    vp = temp_matrix[:, i]
                    vp1 = temp_matrix[:, i + 1]

                    marker.points.append(Point(vp[0], vp[1], 0))
                    marker.points.append(Point(vp1[0], vp1[1], 0))

                marker.id = obs_index
                marker_array.markers.append(marker)

            else:
                marker.type = marker.CYLINDER

                center = obs.center
                marker.scale.x = obs.radius * 2
                marker.scale.y = obs.radius * 2
                marker.scale.z = 0.2

                marker.pose.position.x = center[0, 0]
                marker.pose.position.y = center[1, 0]

                marker.id = obs_index

                marker_array.markers.append(marker)

        return marker_array

    def convert_to_path(self, state_list):
        # from state list to path
        path = Path()

        path.header.seq = 0
        path.header.stamp = self.get_clock().now()
        path.header.frame_id = self.target_frame

        for i in range(len(state_list)):
            ps = PoseStamped()

            ps.header.seq = i
            ps.header.stamp = self.get_clock().now()
            ps.header.frame_id = self.target_frame

            ps.pose.position.x = state_list[i][0, 0]
            ps.pose.position.y = state_list[i][1, 0]
            ps.pose.orientation.w = 1

            path.poses.append(ps)

        return path

    def convert_to_twist(self, rda_vel):
        # from 2*1 vector to twist

        vel = Twist()
        vel.linear.x = rda_vel[0, 0]  # linear
        vel.angular.z = rda_vel[1, 0]  # steering

        return vel

    @staticmethod
    def quat_to_yaw(quater):

        x = quater.x
        y = quater.y
        z = quater.z
        w = quater.w

        raw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return raw

    @staticmethod
    def quat_to_yaw_list(quater):

        x = quater[0]
        y = quater[1]
        z = quater[2]
        w = quater[3]

        raw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return raw

    def generate_robot_tuple(self, robot_info):

        if robot_info is None:
            print("Lack of car information, please check the robot_info in config file")
            return

        if (
            robot_info.get("vertices", None) is None
            or robot_info.get("vertices", None) == "None"
        ):
            length = robot_info["length"]
            width = robot_info["width"]
            wheelbase = robot_info["wheelbase"]

            start_x = -(length - wheelbase) / 2
            start_y = -width / 2

            point0 = np.array([[start_x], [start_y]])  # left bottom point
            point1 = np.array([[start_x + length], [start_y]])
            point2 = np.array([[start_x + length], [start_y + width]])
            point3 = np.array([[start_x], [start_y + width]])

            vertex = np.hstack((point0, point1, point2, point3))

            G, h = self.generate_Gh(vertex)
        else:
            G, h = self.generate_Gh(robot_info["vertices"])

        cone_type = robot_info["cone_type"]
        max_speed = robot_info["max_speed"]
        max_acce = robot_info["max_acce"]
        dynamics = robot_info["dynamics"]

        robot_info_tuple = robot_tuple(
            G, h, cone_type, wheelbase, max_speed, max_acce, dynamics
        )

        return robot_info_tuple

    def generate_Gh(self, vertex):
        """
        vertex: 2*num
        """

        num = vertex.shape[1]

        G = np.zeros((num, 2))
        h = np.zeros((num, 1))

        for i in range(num):
            if i + 1 < num:
                pre_point = vertex[:, i]
                next_point = vertex[:, i + 1]
            else:
                pre_point = vertex[:, i]
                next_point = vertex[:, 0]

            diff = next_point - pre_point

            a = diff[1]
            b = -diff[0]
            c = a * pre_point[0] + b * pre_point[1]

            G[i, 0] = a
            G[i, 1] = b
            h[i, 0] = c

        return G, h

    def get_transform(self, state):
        """
        Get rotation and translation matrices from state.

        Args:
            state (np.array): State [x, y, theta] (3x1) or [x, y] (2x1).

        Returns:
            tuple: Translation vector and rotation matrix.
        """

        if state.shape == (2, 1):
            rot = np.array([[1, 0], [0, 1]])
            trans = state[0:2]
        else:
            rot = np.array(
                [
                    [cos(state[2, 0]), -sin(state[2, 0])],
                    [sin(state[2, 0]), cos(state[2, 0])],
                ]
            )
            trans = state[0:2]
        return trans, rot

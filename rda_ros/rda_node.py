#! /usr/bin/env python
import rclpy

from rda_ros.rda_core import rda_core


def main(args=None):
    rclpy.init(args=args)

    rda = rda_core()
    rda.control()

    rda.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

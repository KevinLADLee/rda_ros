import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rda_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),  # Example for config files
    ],
    install_requires=['setuptools', 'rclpy', 'std_msgs'],  # Add other dependencies here
    zip_safe=True,
    maintainer='invs',
    maintainer_email='kevinladlee@gmail.com',
    description='The ros2 implementation of rda',
    keywords=['ROS'],
    license='GPLv2',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rda_node = rda_ros.rda_node:main'
        ],
    },
)

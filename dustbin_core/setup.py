from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dustbin_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Diya Dutta',
    maintainer_email='diya19012008@gmail.com',
    description='ROS 2 package for Droid Dustbin',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_detector_node = dustbin_core.arm_detector_node:main',
            'lid_controller_node = dustbin_core.lid_controller_node:main',
        ],
    },
)

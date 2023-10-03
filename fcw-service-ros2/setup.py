from setuptools import setup

package_name = 'fcw_service_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Petr Kleparnik',
    maintainer_email='p.kleparnik@cognitechna.cz',
    description='Early collision warning Network Application for Transportation use case - ROS 2 service',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fcw_service_node = fcw_service_ros2.fcw_service_node:main',

        ],
    },
)

from setuptools import setup

package_name = 'client_ros2'

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
    maintainer_email='ikleparnik@fit.vutbr.cz',
    description='CollisionWarningService client',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'client_node = client_ros2.client_node:main',

        ],
    },
)

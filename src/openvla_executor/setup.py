from setuptools import find_packages, setup

package_name = 'openvla_executor'

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
    maintainer='zhaoxinyu',
    maintainer_email='zhaoxinyu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'action_listener = openvla_executor.action_listener:main',
            'delta_executor = openvla_executor.delta_executor:main',
            'pose_executor = openvla_executor.pose_executor:main',
        ],
    },
)

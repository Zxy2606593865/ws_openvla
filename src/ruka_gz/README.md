# ROS 2 jazzy

locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale  # verify settings

sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update && sudo apt install ros-dev-tools

sudo apt update

sudo apt upgrade

sudo apt install ros-jazzy-desktop

echo 'source  /opt/ros/jazzy/setup.bash' >> ~/.bashrc

# ROS WS

sudo apt update && rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y

mkdir -p ~/ws_moveit/src

cd ~/ws_moveit
colcon build

echo 'source ~/ws_moveit/install/setup.bash' >> ~/.bashrc


# GZ Harmonic

sudo apt-get update
sudo apt-get install lsb-release gnupg

sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic

# ros2 control

sudo apt install ros-jazzy-ros2-control

sudo apt install ros-jazzy-gz-ros2-control

sudo apt install ros-jazzy-ros2-controllers

sudo apt-get install ros-${ROS_DISTRO}-ros-gz


# MoveIT2

sudo apt install ros-jazzy-moveit

sudo apt install ros-jazzy-moveit-planners-chomp

sudo apt-get install ros-jazzy-imu-tools

sudo apt-get install can-utils


# PKG
~/ws_moveit/src

git clone https://github.com/VB-Industrial/ruka_gz.git

colcon build --packages-select ruka_gz

# launch 

ros2 launch ruka_gz gazebo.launch.py

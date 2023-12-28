# 数字手势识别小游戏

## 数字手势识别小游戏规则说明

每局10个数字总计10分，答题时间一共2s，在1秒内反应出正确答案则记1分
,1-1.5s记0.5分，1.5-2s记0分

|    时间    | 分数 |
|:--------:|:--:|
|   1.0s   | 1  |
| 1.0-1.5s |0.5 |
|  1.5-2s  | 0  |

在设置界面设置，对应颜色，和手势可设置。
在测试界面，除刷新颜色（红、绿）、以及摄像头框外无其他内容。
数字是绿色的，那就比数字对应的手势；数字是红色的就比除了这个数字的手势。

## 数字手势识别小游戏展示视频

<video width="480" height="360" controls>
  <source src="./doc/数字手势识别小游戏.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 环境配置

### `PyTorch` 安装

#### 从官网下载 `whl` 文件

[PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

进入下载目录以后按 Nvidia 照官网指南安装

```bash
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev
export TORCH_INSTALL=path/to/torch-x.x.x+nvxx.xx-cp38-cp38-linux_aarch64.whl
python3 -m pip install --upgrade pip
python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3'
export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"
python3 -m pip install --upgrade protobuf
python3 -m pip install --no-cache $TORCH_INSTALL
```

#### 从源码编译 (Optional)

参考此链接：[从源码编译Pytorch](./doc/从源码编译PyTorch/从源码编译PyTorch.md)

### `OpenCV` GPU 编译版 (Optional)

参考此链接：

### `PyQt5` 安装

```bash
pip install PyQt5
```

如果在 Jetson 中使用了虚拟环境则需要从源码编译，参考此链接：[Jetson 安装 PyQt5](https://blog.csdn.net/qq_41893274/article/details/104103622)

### `MediaPipe` 安装

#### CPU 编译版本

```bash
pip install mediapipe
```

#### GPU 编译版本 (Optional)

参考此链接：[GPU编译Mediaipe](./doc/GPU编译MediaPipe/GPU编译MediaPipe.md)

### 其他依赖项安装

```bash
pip install -r requirements.txt
```

## 运行小游戏

```bash
python3 main.py
```

# 机械臂手势跟随

## 机械臂手势跟随展示视频

<video width="480" height="360" controls>
  <source src="./doc/机械臂跟随手指手势运动.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 环境配置

### `ROS` 安装

```bash
wget http://fishros.com/install -O fishros && bash fishros
```

借助 ROS 社区鱼香肉丝大佬的脚本安装ROS，按照提示安装 `Noetic` 版本即可。

### `MoveIt` 安装

```bash
sudo apt-get install ros-noetic-moveit
```

### MoveIt 相关控制器安装

```bash
sudo apt-get install ros-noetic-*-controller
```

## 创建 ROS 工作空间

创建一个 ROS 工作空间

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
```

将 `jibot3` 和 `jibot3_moveit_config` 包放入 `~/catkin_ws/src` 目录下

```bash
cd ~/catkin_ws
catkin_make
```

## 运行机械臂手势跟随

1. 首先 source ROS 环境

    ```bash
    source /opt/ros/noetic/setup.bash
    source ~/catkin_ws/devel/setup.bash
    ```

2. 启动 `roscore`

    ```bash
    roscore
    ```

3. 启动 `jibot3` 机械臂仿真环境和 `MoveIt` 控制器

    ```bash
    roslaunch jibot3_moveit_config demo.launch
    ```

4. 启动 `jibot3` 机械臂手势跟随节点

    ```bash
    cd path/to/Gesture_Recognition
    python3 gesture_node.py
    ```
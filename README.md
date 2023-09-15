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

# 步骤
## 安装环境
### 安装anaconda

### pip换源
```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
### 安装包
```SHELL
pip install opencv-python mediapipe PyQt5 PyQt5-tools
```
### 添加环境变量

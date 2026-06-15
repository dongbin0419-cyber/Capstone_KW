# UR3 + Schunk + RealSense D455 통합 실행 README

---

## 1. 기본 환경

권장 환경:

- Ubuntu 24.04
- ROS 2 Jazzy
- opencv-python<4.10
- numpy<2.0.0
- MoveIt 2

---

## 2. 필요한 패키지 설치

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-moveit \
  ros-jazzy-ur \
  ros-jazzy-ur-robot-driver \
  ros-jazzy-realsense2-camera \
  ros-jazzy-xacro \
  ros-jazzy-tf2-ros \
  ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-rviz2 \
  python3-pip
```

파이썬 라이브러리:

```bash
pip install ultralytics opencv-python open3d scipy numpy
```

설치 확인:

```bash
ros2 --version
python3 -c "import cv2, open3d, ultralytics, scipy, numpy; print('python libs OK')"
```

---

## 3. 워크스페이스 구조

다른 PC에서도 아래 구조를 맞추는 것이 가장 편합니다.

```bash
/home/dongbin/vs_ws/
├── src/
│   ├── ur3_control/
│   │   └── urdf/
│   │       └── ur3_robot.urdf.xacro
│   ├── ur3_moveit_config_custom/
│   │   └── launch/
│   │       └── move_group.launch.py
│   └── pointcloud_bottle_all.py
├── build/
├── install/
└── log/
```

코드 안에서 기본 경로가 아래처럼 잡혀 있습니다.

```python
WS_SETUP = "/home/dongbin/vs_ws/install/setup.bash"
URDF = "/home/dongbin/vs_ws/src/ur3_control/urdf/ur3_robot.urdf.xacro"
```

다른 PC 사용자명이 다르면 코드의 이 경로를 바꿔야 합니다.

예를 들어 사용자명이 `robot`이면:

```python
WS_SETUP = "/home/robot/vs_ws/install/setup.bash"
URDF = "/home/robot/vs_ws/src/ur3_control/urdf/ur3_robot.urdf.xacro"
```

---

## 4. 커스텀 URDF 준비

커스텀 URDF 파일 위치:

```bash
/home/dongbin/vs_ws/src/ur3_control/urdf/ur3_robot.urdf.xacro
```

이 파일에는 기본 UR3뿐 아니라 다음 내용이 포함되어 있어야 합니다.

- UR3 링크/조인트
- 카메라 링크 또는 카메라 TF 기준
- Schunk 그리퍼 링크
- `gripper_tcp` 링크
- `base_link` 기준 TF 연결

코드에서 핵심으로 쓰는 프레임:

```python
self.base_frame = "base_link"
self.gripper_frame = "gripper_tcp"
```

따라서 URDF 안에 `gripper_tcp`가 반드시 있어야 합니다.

---

## 5. 커스텀 URDF 단독 확인

URDF가 제대로 파싱되는지 먼저 확인합니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/vs_ws/install/setup.bash

ros2 run xacro xacro ~/vs_ws/src/ur3_control/urdf/ur3_robot.urdf.xacro > /tmp/ur3_robot.urdf
check_urdf /tmp/ur3_robot.urdf
```


## 6. 커스텀 URDF 퍼블리시

코드 안에서는 아래 명령을 자동으로 실행합니다.

```bash
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args \
  -p robot_description:="$(ros2 run xacro xacro /home/dongbin/vs_ws/src/ur3_control/urdf/ur3_robot.urdf.xacro)"
```

수동으로 확인하려면:

```bash
source /opt/ros/jazzy/setup.bash
source ~/vs_ws/install/setup.bash

ros2 run robot_state_publisher robot_state_publisher \
  --ros-args \
  -p robot_description:="$(ros2 run xacro xacro ~/vs_ws/src/ur3_control/urdf/ur3_robot.urdf.xacro)"
```

다른 터미널에서 TF 확인:

```bash
ros2 run tf2_ros tf2_echo base_link gripper_tcp
```

정상이라면 `base_link -> gripper_tcp` 변환값이 계속 나옵니다.

---

## 7. MoveIt 커스텀 패키지

코드에서 사용하는 MoveIt launch:

```bash
ros2 launch ur3_moveit_config_custom move_group.launch.py
```

따라서 다른 PC에도 아래 패키지가 있어야 합니다.

```bash
~/vs_ws/src/ur3_moveit_config_custom
```

확인:

```bash
source /opt/ros/jazzy/setup.bash
source ~/vs_ws/install/setup.bash

ros2 pkg list | grep ur3_moveit_config_custom
```

안 나오면 빌드가 안 된 것입니다.

---

## 8. 워크스페이스 빌드

```bash
cd ~/vs_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

빌드 후 확인:

```bash
ros2 pkg list | grep ur3_control
ros2 pkg list | grep ur3_moveit_config_custom
```

---

## 9. 네트워크 설정

현재 코드 기준:

```python
ROBOT_IP = "192.168.1.101"
reverse_ip = "192.168.1.102"
```

즉 PC의 유선 IP를 `192.168.1.102`로 맞춰야 합니다.

Ubuntu 네트워크 설정 예시:

```bash
IP address: 192.168.1.102
Netmask: 255.255.255.0
Gateway: 비워도 됨
```

연결 확인:

```bash
ping 192.168.1.101
```

응답이 와야 합니다.

---

## 10. UR 티치펜던트 준비

UR 로봇에는 External Control 프로그램이 있어야 합니다.

코드에서 자동 실행하는 명령:

```python
send_dashboard_command("load External_Control.urp")
send_dashboard_command("play")
```

티치펜던트에서 > 좌측 상단에 프로그램 > UR CAP(이었나?) 클릭 > External_Control 클릭

문제 생기면 수동으로:
1. 티치펜던트에서 > 좌측 상단에 프로그램 > UR CAP(이었나?) 클릭 > External_Control 클릭
2. Play 누르기

---

## 11. RealSense 확인

카메라 연결 확인:

```bash
rs-enumerate-devices
```

ROS 토픽 확인:

```bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  pointcloud.enable:=true \
  pointcloud.ordered_pc:=true \
  publish_tf:=false
```

다른 터미널:

```bash
ros2 topic list | grep camera
```

필수 토픽:

```bash
/camera/camera/color/image_raw/compressed
/camera/camera/depth/color/points
/camera/camera/depth/camera_info
```

---

## 12. 코드 파일 위치

권장 위치:

```bash
~/vs_ws/src/pointcloud_bottle_all.py
```

실행 권한 부여:

```bash
chmod +x ~/vs_ws/src/pointcloud_bottle_all.py
```

---

## 13. 실행 명령

코드를 먼저 실행한 뒤, 다른 노드나 터미널에서 /grasp_command와 /approach_done을 보내면 동작을 시작합니다.

기본 실행:

cd ~/vs_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

python3 src/pointcloud_bottle_all.py

실행 후 코드는 아래 두 신호를 기다립니다.

/grasp_command  : String
/approach_done  : Bool
---

## 13-1
예시 1. 책상 + 서있는 물병:

ros2 topic pub --once /grasp_command std_msgs/msg/String "{data: 'table standing bottle'}"
ros2 topic pub --once /approach_done std_msgs/msg/Bool "{data: true}"

예시 2. 침대 + 서있는 물병:

ros2 topic pub --once /grasp_command std_msgs/msg/String "{data: 'bed standing bottle'}"
ros2 topic pub --once /approach_done std_msgs/msg/Bool "{data: true}"

예시 3. 책상 + 컵:

ros2 topic pub --once /grasp_command std_msgs/msg/String "{data: 'table standing cup'}"
ros2 topic pub --once /approach_done std_msgs/msg/Bool "{data: true}"

/grasp_command 형식은 반드시 아래처럼 띄어쓰기 3개 값입니다.

scene pose object

예:

table standing bottle
bed standing cup

## 13-2
작업이 끝나고 복귀자세까지 이동하면 코드가 아래 토픽을 publish합니다.

/grasp_done : Bool

완료 확인:

ros2 topic echo /grasp_done

정상 완료 시:

data: true

## 14. 코드 실행 순서

pointcloud_bottle_all.py는 내부적으로 아래 순서로 실행됩니다.

1. 기존 ROS/UR/RealSense 프로세스 강제 종료
2. ROS daemon 재시작
3. RealSense 실행
4. UR driver 실행
5. Dashboard로 External_Control.urp 로드
6. Dashboard로 play
7. 커스텀 URDF robot_state_publisher 실행
8. 커스텀 MoveIt move_group 실행
9. 컨트롤러를 scaled_joint_trajectory_controller로 전환
10. /grasp_command 대기
11. 명령에 맞는 초기자세로 이동
12. /approach_done=True 대기
13. 두 신호가 모두 오면 YOLO + PointCloud 인식 시작
14. MoveGroup 접근
15. wrist 정렬
16. YOLO X 보정
17. 직선 전진
18. 그리퍼 close
19. 5cm lift
20. 통일된 fixed 복귀자세로 이동
21. /grasp_done=True publish
22. 코드 종료


## 15. 모드 설정 위치

코드 상단의 `CONFIG`에서 모드를 관리합니다.

```python
CONFIG = {
    "bed": {
        "standing": {
            "bottle": {...},
            "cup": {...},
            "remote": {...},
        },
        "lying": {
            "bottle": {...},
            "cup": {...},
            "remote": {...},
        },
    },
    "table": {
        "standing": {
            "bottle": {...},
            "cup": {...},
            "remote": {...},
        },
        "lying": {
            "bottle": {...},
            "cup": {...},
            "remote": {...},
        },
    },
}
```


## 16. 초기자세 변경 위치
초기자세는 bed/table에 따라 다릅니다.

BED_INITIAL_POSE = [...]
TABLE_INITIAL_POSE = [...]

명령이 들어오면 해당 모드에 맞는 초기자세로 먼저 이동합니다.

복귀자세는 모든 모드에서 통일합니다.

return_pose = [
    4.558644771575928,
    -0.13444264352832036,
    -1.6154195070266724,
    -1.8358494244017542,
    1.6432443857192993,
    -1.6180499235736292
]

즉,

초기자세: bed/table 별도
복귀자세: 전부 동일 (복귀자세 변경필요)


## 17. YOLO 모델 변경

모델 변경은 실행 명령에서 --model로 넣는 방식이 아니라, 코드 상단 CONFIG 내부에서 변경합니다.

기본 모델:

"model_path": "yolov8n.pt"

커스텀 모델 예시:

"model_path": "/home/dongbin/vs_ws/src/models/remote.pt"
YOLO 기본 모델에서 인식되지 않는 물체는 별도 학습한 .pt 모델을 model_path에 넣어야 합니다.


## 18. 마커와 보정값

빨간 구:

```python
publish_target_marker()
```

로봇이 잡으러 갈 목표점을 표시합니다.

목표점 보정은 CONFIG의 `offset_x`, `offset_y`로 조정합니다.

```python
"offset_x": -0.05,
"offset_y": 0.00,
```

코드 내부 적용:

```python
centroid_cam[0] -= self.offset_x
centroid_cam[1] -= self.offset_y
```

현재 `offset_x = -0.05`이면 실제로는 `centroid_cam[0]`에 `+0.05m`가 들어가는 효과입니다.

---

## 19. Z Search

초기 카메라 켜진 직후 이상 상승 방지용:

```python
self.z_search_start_delay = 10.0
```

의미:

- 노드 생성 후 10초 동안은 Z Search 금지
- YOLO/PointCloud가 순간적으로 안 잡혀도 로봇이 위아래로 움직이지 않음

너무 오래 기다리면 줄이면 됩니다.


## 20. 경로 생성이 오래 걸릴 때

MoveGroup 설정 위치:

```python
request.num_planning_attempts = 3
request.allowed_planning_time = 4.0
sphere.dimensions = [0.08]
```

빠르게 실패하게 하고 싶으면:

```python
request.num_planning_attempts = 1
request.allowed_planning_time = 2.0
```

성공률을 높이고 싶으면:

```python
request.num_planning_attempts = 5
request.allowed_planning_time = 8.0
```

대신 오래 걸립니다.

---

## 21. 자주 나는 에러

### 21-1. MoveGroup failed error_code=99999

대부분 원인:

- 목표점이 너무 멀다
- 목표 z가 너무 낮다
- orientation constraint가 빡세다
- joint constraint 때문에 샘플링 가능한 goal state가 없다
- `gripper_tcp` TF가 실제와 다르다

확인:

```bash
ros2 run tf2_ros tf2_echo base_link gripper_tcp
ros2 topic echo /detected_bottle_pose
```

---

### 21-2. Unable to sample any valid states for goal tree

해결 후보:

```python
sphere.dimensions = [0.08]  # 0.06보다 여유
ori_constraint.absolute_x_axis_tolerance = 1.5
ori_constraint.absolute_y_axis_tolerance = 1.5
ori_constraint.absolute_z_axis_tolerance = 3.14
```

그래도 안 되면 목표점 z를 너무 낮게 잡는지 확인합니다.

---


### 21-3. 리얼센스 카메라 튕김
화면이 멈춘후, 터미널에 tf가 끊겼다는 주황색 로그 뜨면 다시 코드 실행해야합니다

### 21-4. Ur3 연결시 aborted
쓰레기 노드나 프로세스들 많이 쌓이면 연결시 aborted 발생합니다. 
정리해주고 로봇 본체 재부팅 해줍니다


## 22. 실행 전 체크리스트

```bash
ping 192.168.1.101
```

```bash
source /opt/ros/jazzy/setup.bash
source ~/vs_ws/install/setup.bash
```

```bash
ros2 pkg list | grep ur3_moveit_config_custom
```

```bash
ros2 run xacro xacro ~/vs_ws/src/ur3_control/urdf/ur3_robot.urdf.xacro > /tmp/ur3_robot.urdf
check_urdf /tmp/ur3_robot.urdf
```

```bash
rs-enumerate-devices
```

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('YOLO OK')"
```

---

## 23. 가상 실행

가장 기본 실행:

cd ~/vs_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

python3 src/pointcloud_bottle_all.py

실행 후 다른 터미널에서 명령을 보냅니다.

책상 + 서있는 물병:

ros2 topic pub --once /grasp_command std_msgs/msg/String "{data: 'table standing bottle'}"
ros2 topic pub --once /approach_done std_msgs/msg/Bool "{data: true}"

침대 + 서있는 물병:

ros2 topic pub --once /grasp_command std_msgs/msg/String "{data: 'bed standing bottle'}"
ros2 topic pub --once /approach_done std_msgs/msg/Bool "{data: true}"

완료 확인:

ros2 topic echo /grasp_done


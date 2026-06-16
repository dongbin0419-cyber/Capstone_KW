import os
import threading
import time
import subprocess
import signal
import atexit
import cv2
import numpy as np
import open3d as o3d
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import socket
from std_msgs.msg import Bool, String
from scipy.spatial.transform import Rotation
from ultralytics import YOLO

from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import CompressedImage, PointCloud2, CameraInfo, JointState
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    PlanningScene,
    CollisionObject,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    MotionPlanRequest,
    PlanningOptions,
    RobotState,
    JointConstraint,
)

from shape_msgs.msg import SolidPrimitive
from ur_msgs.srv import SetIO

import tf2_ros
from tf2_geometry_msgs import do_transform_pose

BED_INITIAL_POSE = [
    4.558537006378174,
    -0.3419717115214844,
    -1.4476462602615356,
    -2.2719394169249476,
    1.6401886940002441,
    -1.6181018988238733,
]

TABLE_INITIAL_POSE = [
    4.558644771575928,
    -0.13444264352832036,
    -1.6154195070266724,
    -1.8358494244017542,
    1.6432443857192993,
    -1.6180499235736292,
]

CONFIG = {
    "bed": {
        "standing": {
            "bottle": {
                "target_object": "bottle",
                "bed_mode": True,
                "initial_pose": BED_INITIAL_POSE,
                "model_path": "yolov8n.pt",
                "offset_x": -0.05,
                "offset_y": 0.00,
                "z_search_start_delay": 10.0,
            },
            "cup": {
                "target_object": "cup",
                "bed_mode": True,
                "initial_pose": BED_INITIAL_POSE,
                "model_path": "yolov8n.pt",
                "offset_x": -0.05,
                "offset_y": 0.00,
                "z_search_start_delay": 10.0,
            },

            # 리모컨 추가 위치
            # "remote": {
            #     "target_object": "remote",
            #     "bed_mode": True,
            #     "initial_pose": BED_INITIAL_POSE,
            #     "model_path": "remote.pt",
            #     "offset_x": -0.05,
            #     "offset_y": 0.00,
            #     "z_search_start_delay": 10.0,
            # },
        }
    },
    "table": {
        "standing": {
            "bottle": {
                "target_object": "bottle",
                "bed_mode": False,
                "initial_pose": TABLE_INITIAL_POSE,
                "model_path": "yolov8n.pt",
                "offset_x": -0.05,
                "offset_y": 0.00,
                "z_search_start_delay": 10.0,
            },
            "cup": {
                "target_object": "cup",
                "bed_mode": False,
                "initial_pose": TABLE_INITIAL_POSE,
                "model_path": "yolov8n.pt",
                "offset_x": -0.05,
                "offset_y": 0.00,
                "z_search_start_delay": 10.0,
            },

            # 리모컨 추가 위치
        }
    }
}

class PointCloudRansacTest(Node):
    def __init__(self, config):
        super().__init__('pc_ransac_bottle_table_test')

        self.config = config
        self.target_object = config["target_object"]
        self.bed_mode = config["bed_mode"]
        self.initial_pose = config["initial_pose"]
        self.offset_x = config["offset_x"]
        self.offset_y = config["offset_y"]
        self.z_search_start_delay = config["z_search_start_delay"]

        self._cloud_lock = threading.RLock()
        self.show_visualization = os.environ.get('DISPLAY') is not None
        self.camera_y_offset_px = 0
        self.pixel_tolerance_y = 20

        try:
            self.model = YOLO(config["model_path"], task='detect')
        except Exception as exc:
            self.get_logger().error(f'YOLO 모델 로딩 실패: {exc}')
            raise

        self.grasp_done_pub = self.create_publisher(
            Bool,
            '/grasp_done',
            10
        )

        self.approach_done_sub = self.create_subscription(
            Bool,
            '/approach_done',
            self.approach_done_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/grasp_command',
            self.grasp_command_callback,
            10
        )
        self.waiting_start = True

        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.servo_pub = self.create_publisher(
            TwistStamped,
            '/servo_node/delta_twist_cmds',
            10
        )
        self.node_start_time = time.time()
        self.target_ever_locked = False
        self.search_z_busy = False
        self.search_z_dir = 1.0
        self.search_z_wait_sec = 2.0
        self.search_z_step = 0.05   # 5cm
        self.scene_initialized = False
        self.camera_x_offset_px = -25  # 나중에 잰 값 넣기. 예: +35, -20
        self.advance_done = False
        self.advancing = False
        self.start_tracking_timer = None
        self.tracking_mode = False
        self.tracking_done = False
        self.pixel_tolerance = 20
        self.camera_info_logged = False
        self.cloud_msg = None
        self.camera_info = None
        self.command_ready = False
        self.approach_ready = False
        self.cartesian_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path'
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/depth/camera_info',
            self.camera_info_callback,
            10
        )

        self.io_client = self.create_client(
            SetIO,
            '/io_and_status_controller/set_io'
        )

        while not self.io_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_io service waiting...')

        self.gripper_open()

        # TF2 버퍼 및 리스너 초기화
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'
        )

        self.is_planning = False
        self.last_plan_time = 0.0
        self.plan_interval = 1.0
        self.goal_active = False
        self.motion_done = False

        # RViz 시각화용 퍼블리셔 등록
        self.table_pc_pub = self.create_publisher(PointCloud2, '/viz/table_points', 10)
        self.bottle_pc_pub = self.create_publisher(PointCloud2, '/viz/bottle_points', 10)
        self.marker_pub = self.create_publisher(Marker, '/viz/markers', 10)

        # UR3 로봇팔 전송용 경로 퍼블리셔
        self.joint_traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        self.tcp_pose_sub = self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.tcp_pose_callback, 10)

        self.current_tcp_pos = None
        self.target_pose_pub = self.create_publisher(
            PoseStamped,
            '/detected_bottle_pose',
            10
        )

        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        self.color_sub = self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.color_callback, 10)
        self.pc_sub = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)

        # 계산된 물병 위치와 크기를 임시 저장할 주머니(변수) 생성
        self.latest_bottle_base = None
        self.latest_bottle_size = None
        self.latest_main_axis_base = None
        self.latest_bottle_pose_type = "standing"
        self.target_lock_need = 5
        self.target_lock_buffer = []
        self.target_locked = False
        self.locked_bottle_base = None
        self.target_lock_spread_limit = 0.03  # 3cm

        # [변경] 비동기 중복 요청 렉 방지를 위해 제어 주기를 2.5초로 여유롭게 늘림
        self.base_frame = 'base_link'
        self.gripper_frame = 'gripper_tcp'


        self.control_timer = self.create_timer(2.5, self.robot_control_loop)
        
        # 켜지자마자 제멋대로 초기 자세로 튀는 현상을 방지하기 위해 완전 봉쇄
        # self.initial_joint_angles = [4.558644771575928, -0.13444264352832036, -1.6154195070266724, -1.8358494244017542, 1.6432443857192993, -1.5688241163836878]
        # self.initial_pose_sent = False
        # self.create_timer(1.0, self.send_initial_pose_once)
        
        self.get_logger().info("PointCloud RANSAC & UR3 Waypoint Generator Node Started")

    def pc_callback(self, msg):
        with self._cloud_lock:
            self.cloud_msg = msg

    def initial_pose_done_once(self):
        if hasattr(self, "initial_pose_timer") and self.initial_pose_timer is not None:
            self.initial_pose_timer.cancel()
            self.initial_pose_timer = None

        self.get_logger().warn("초기자세 이동 완료 → grasp 시작 조건 확인")
        self.try_start_grasp()

    def try_start_grasp(self):
        if not self.waiting_start:
            return

        if self.command_ready and self.approach_ready:
            self.waiting_start = False
            self.node_start_time = time.time()
            self.get_logger().warn("명령 + /approach_done 수신 완료 → grasp 시작")

    def approach_done_callback(self, msg):
        if not msg.data:
            return

        self.approach_ready = True

        self.get_logger().warn("/approach_done 수신 완료")

        self.try_start_grasp()

    def grasp_command_callback(self, msg):
        try:
            scene, pose, obj = msg.data.strip().split()

            self.config = CONFIG[scene][pose][obj].copy()
            self.target_object = self.config["target_object"]
            self.bed_mode = self.config["bed_mode"]
            self.initial_pose = self.config["initial_pose"]
            self.offset_x = self.config["offset_x"]
            self.offset_y = self.config["offset_y"]
            self.z_search_start_delay = self.config["z_search_start_delay"]

            model_path = self.config["model_path"]
            self.model = YOLO(model_path, task="detect")

            self.command_ready = True

            self.get_logger().warn(
                f"/grasp_command 수신: scene={scene}, pose={pose}, object={obj}"
            )

            self.send_initial_joint_pose(self.initial_pose)

            self.initial_pose_timer = self.create_timer(
                21.0,
                self.initial_pose_done_once
            )

        except Exception as e:
            self.get_logger().error(
                f"grasp_command 형식 오류. 예: 'table standing bottle' / error={e}"
            )

    def send_initial_joint_pose(self, initial_pose):
        joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        traj = JointTrajectory()
        traj.joint_names = joint_order

        point = JointTrajectoryPoint()
        point.positions = initial_pose
        point.time_from_start.sec = 20

        traj.points.append(point)
        self.joint_traj_pub.publish(traj)

        self.get_logger().warn("명령에 맞는 초기자세 이동 시작")

    
    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

        if not self.camera_info_logged:
            self.get_logger().info(
                f"CameraInfo received: "
                f"fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, "
                f"cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}"
            )
            self.camera_info_logged = True

    def tcp_pose_callback(self, msg):
        self.current_tcp_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)


    def align_gripper_y_to_bottle_axis_by_wrist3(self):
        if self.latest_main_axis_base is None:
            self.get_logger().warn("물병 긴축 정보 없음")
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            q = tf.transform.rotation
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
            R = rot.as_matrix()

            x_axis = R[:, 0]      # wrist3 회전축 / 전진축
            jaw_axis = R[:, 1]  # gripper_tcp 벌어지는 방향

            obj_axis = np.array(self.latest_main_axis_base, dtype=np.float64)
            obj_axis = obj_axis / np.linalg.norm(obj_axis)

            # wrist3는 TCP X축 기준 회전이라고 보고,
            # 현재 y_axis를 x_axis 주변으로 돌려서 obj_axis와 수직이 되게 함
            a = np.dot(jaw_axis, obj_axis)
            b = np.dot(np.cross(x_axis, jaw_axis), obj_axis)

            angle = np.arctan2(b, a)

            # 180도 크게 돌지 말고 -90~90도 안에서만 보정
            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi

            self.get_logger().warn(
                f"ALIGN BOTTLE AXIS: dot(z,obj)={a:.3f}, "
                f"wrist3 correction={np.degrees(angle):.1f} deg"
            )

            self.rotate_wrist3_only(angle)

        except Exception as e:
            self.get_logger().warn(f"align_gripper_y_to_bottle_axis failed: {e}")

    def transform_vector_to_base(self, cloud_msg, vec_cam, target_frame=None):
        source_frame = cloud_msg.header.frame_id.strip('/')

        if target_frame is None:
            target_frame = self.get_robot_base_frame()

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            q = transform.transform.rotation
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w])

            vec_base = rot.apply(vec_cam)
            vec_base = vec_base / np.linalg.norm(vec_base)

            return vec_base

        except Exception as e:
            self.get_logger().warn(
                f"Vector TF failed: {source_frame} -> {target_frame}, {e}"
            )
            return None
        
    def correct_axis_perspective(self, axis_img, center_cam):
        x = center_cam[0]
        y = center_cam[1]
        z = center_cam[2]

        # 카메라가 +Y 방향을 본다는 가정
        yaw = np.arctan2(x, y)     # 좌우 치우침
        pitch = np.arctan2(z, y)   # 상하 치우침

        # 화면상 기울어져 보이는 방향을 반대로 보정
        correction = -pitch

        c = np.cos(correction)
        s = np.sin(correction)

        R2 = np.array([
            [c, -s],
            [s,  c]
        ])

        axis_corr = R2 @ axis_img
        axis_corr = axis_corr / np.linalg.norm(axis_corr)

        return axis_corr

    def rotate_wrist3_only(self, delta_rad):
        if self.current_joint_state is None:
            self.get_logger().warn("joint_state 없음")
            return

        joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        pos_dict = dict(zip(
            self.current_joint_state.name,
            self.current_joint_state.position
        ))

        target_pos = [float(pos_dict[name]) for name in joint_order]

        # 여기만 바뀜
        target_pos[5] += float(delta_rad)

        traj = JointTrajectory()
        traj.joint_names = joint_order

        point = JointTrajectoryPoint()
        point.positions = target_pos
        point.time_from_start.sec = 2

        traj.points.append(point)
        self.joint_traj_pub.publish(traj)

        self.get_logger().warn(
            f"ONLY wrist_3_joint rotated: {np.degrees(delta_rad):.1f} deg"
        )

    def send_forward_cartesian_path(self, target_pose):
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("compute_cartesian_path service 없음")
            self.advancing = False
            return

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.group_name = "ur_manipulator"
        req.link_name = self.gripper_frame

        if self.current_joint_state is not None:
            req.start_state = RobotState()
            req.start_state.joint_state = self.current_joint_state
            req.start_state.is_diff = False

        req.waypoints.append(target_pose.pose)
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self.cartesian_path_callback)


    def cartesian_path_callback(self, future):
        try:
            res = future.result()

            if res.fraction < 0.9:
                self.get_logger().warn(f"Cartesian path 실패: fraction={res.fraction:.2f}")
                self.advancing = False
                return

            traj = res.solution.joint_trajectory
            traj.header.stamp = self.get_clock().now().to_msg()

            speed_scale = 3.0   # 숫자 클수록 느려짐. 2~5 추천

            for p in traj.points:
                p.velocities = []
                p.accelerations = []
                p.effort = []

                total_ns = (
                    p.time_from_start.sec * 1_000_000_000
                    + p.time_from_start.nanosec
                )
                total_ns = int(total_ns * speed_scale)
                p.time_from_start.sec = total_ns // 1_000_000_000
                p.time_from_start.nanosec = total_ns % 1_000_000_000

            self.joint_traj_pub.publish(traj)

            self.get_logger().warn(f"Cartesian 직선 전진 실행: fraction={res.fraction:.2f}")

            last_t = traj.points[-1].time_from_start
            done_time = last_t.sec + last_t.nanosec * 1e-9 + 0.5

            self.forward_done_timer = self.create_timer(
                done_time,
                self.forward_cartesian_done_once
            )

        except Exception as e:
            self.get_logger().warn(f"Cartesian path error: {e}")
            self.advancing = False


    def forward_cartesian_done_once(self):
        if hasattr(self, "forward_done_timer") and self.forward_done_timer is not None:
            self.forward_done_timer.cancel()
            self.forward_done_timer = None

        self.advancing = False

        self.get_logger().warn("직선 전진 완료 → gripper close")
        self.gripper_close()

        self.lift_timer = self.create_timer(
            1.0,
            self.lift_then_return_once
        )

    def update_planning_scene_cluster_obstacles(self, table_pts_base, bottle_base):
        from moveit_msgs.msg import PlanningScene, CollisionObject
        from geometry_msgs.msg import Pose
        from shape_msgs.msg import SolidPrimitive
        import numpy as np

        if table_pts_base is None or len(table_pts_base) < 30:
            return

        pts = np.asarray(table_pts_base)

        # 1. 포인트군을 voxel 중심점으로 변환
        voxel_size = 0.03  # 5cm

        voxel_idx = np.floor(pts / voxel_size).astype(np.int32)

        _, unique_indices = np.unique(
            voxel_idx,
            axis=0,
            return_index=True
        )

        voxel_centers = pts[unique_indices]

        # 2. 물병 주변은 비우기
        clear_radius = 0.08  # 물병 주변 15cm 제거

        dist_xy = np.linalg.norm(
            voxel_centers[:, :2] - np.array(bottle_base[:2]),
            axis=1
        )

        voxel_centers = voxel_centers[dist_xy > clear_radius]

        # 3. voxel 개수 제한
        max_voxels = 100

        if len(voxel_centers) > max_voxels:
            step = max(1, len(voxel_centers) // max_voxels)
            voxel_centers = voxel_centers[::step][:max_voxels]

        # 4. 기존 장애물 제거
        clean_scene = PlanningScene()
        clean_scene.is_diff = True

        for obj_id in ["detected_table", "safety_floor"]:
            obj = CollisionObject()
            obj.header.frame_id = self.get_robot_base_frame()
            obj.id = obj_id
            obj.operation = CollisionObject.REMOVE
            clean_scene.world.collision_objects.append(obj)

        self.planning_scene_pub.publish(clean_scene)

        # 5. 새 planning scene 생성
        scene = PlanningScene()
        scene.is_diff = True

        # 6. 테이블 voxel 장애물 생성
        table_obj = CollisionObject()
        table_obj.header.frame_id = self.get_robot_base_frame()
        table_obj.id = "detected_table"
        table_obj.operation = CollisionObject.ADD

        for p in voxel_centers:
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [
                voxel_size,
                voxel_size,
                voxel_size
            ]

            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            pose.orientation.w = 1.0

            table_obj.primitives.append(box)
            table_obj.primitive_poses.append(pose)

        # 7. 바닥 안전 박스 생성
        floor_obj = CollisionObject()
        floor_obj.header.frame_id = self.get_robot_base_frame()
        floor_obj.id = "safety_floor"
        floor_obj.operation = CollisionObject.ADD

        floor_box = SolidPrimitive()
        floor_box.type = SolidPrimitive.BOX
        # 로봇 베이스 주변/뒤쪽만 안전 바닥 박스
        # base_link 기준 X 앞방향이 +X라고 가정
        # X=+0.10m 이후에는 박스 없음
        floor_box.dimensions = [2.0, 1.2, 0.02]

        floor_pose = Pose()
        floor_pose.position.x = 0.0   # 범위: -1.10m ~ +0.10m
        floor_pose.position.y = -0.50
        floor_pose.position.z = -0.05
        floor_pose.orientation.w = 1.0

        floor_obj.primitives.append(floor_box)
        floor_obj.primitive_poses.append(floor_pose)

        # 8. scene publish
        scene.robot_state.is_diff = True
        scene.world.collision_objects.append(table_obj)
        scene.world.collision_objects.append(floor_obj)

        self.planning_scene_pub.publish(scene)

        self.get_logger().info(
            f"군집 장애물 생성 완료: "
            f"{len(voxel_centers)}개 voxel, "
            f"clear_radius={clear_radius:.2f}m"
        )

    def align_gripper_y_parallel_to_floor_by_wrist3(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            q = tf.transform.rotation
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
            R = rot.as_matrix()

            x_axis = R[:, 0]
            jaw_axis = R[:, 1]
            world_z = np.array([0.0, 0.0, 1.0])

            a = np.dot(jaw_axis, world_z)
            b = np.dot(np.cross(x_axis, jaw_axis), world_z)

            angle = np.arctan2(-a, b)

            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi

            self.get_logger().warn(
                f"FLOOR ALIGN: jaw_z={jaw_axis[2]:.3f}, "
                f"wrist3 correction={np.degrees(angle):.1f} deg"
            )

            self.rotate_wrist3_only(angle)

        except Exception as e:
            self.get_logger().warn(f"FLOOR ALIGN 실패: {e}")

    def start_tracking_once(self):
        if self.start_tracking_timer is not None:
            self.start_tracking_timer.cancel()
            self.start_tracking_timer = None


        # 중요: 도착 후 다시 경로계획 못 하게 막기
        self.motion_done = True

        # 도착 후 X 보정 시작
        self.tracking_mode = True
        self.tracking_done = False
        self.goal_active = False

        self.get_logger().warn("WRIST 완료 → YOLO X 보정 시작")


    def send_move_group_goal(self, bottle_base):
        self.get_logger().warn("ENTER send_move_group_goal")
        if self.is_planning or self.current_tcp_pos is None:
            self.get_logger().warn("Planning 중이거나 현재 TCP 위치를 알 수 없어 스킵합니다.")
            return


        self.is_planning = True
        self.last_plan_time = time.time()

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.get_robot_base_frame()
        target_pose.header.stamp = self.get_clock().now().to_msg()

        try:
            p_tcp = self.current_tcp_pos
            p_bottle = np.array([bottle_base[0], bottle_base[1], bottle_base[2]])


            # 물병 바로 위/근처가 아니라, 물병 앞 15cm 접근 포즈
            p_tcp = np.array(self.current_tcp_pos, dtype=np.float64)

            p_bottle = np.array([
                bottle_base[0],
                bottle_base[1],
                bottle_base[2]
            ], dtype=np.float64)

            # 물병 중심을 조금 더 명확히 잡음
            # bbox_center가 이미 중심이면 +0 안 함
            look_point = p_bottle.copy()

            if self.bed_mode:
                look_point[2] = max(look_point[2] + 0.01, 0.05)
            else:
                look_point[2] += 0.01

            # 수평 접근이면 XY 방향만 사용해서 접근점 생성
            approach_xy = look_point[:2] - p_tcp[:2]

            if np.linalg.norm(approach_xy) < 1e-6:
                approach_xy = np.array([1.0, 0.0])
            else:
                approach_xy = approach_xy / np.linalg.norm(approach_xy)

            offset_distance = 0.15

            target_pos = np.array([
                look_point[0] - approach_xy[0] * offset_distance,
                look_point[1] - approach_xy[1] * offset_distance,
                look_point[2]
            ], dtype=np.float64)

            target_pose.pose.position.x = float(target_pos[0])
            target_pose.pose.position.y = float(target_pos[1])
            target_pose.pose.position.z = float(target_pos[2])#0.02M 2cm

            self.last_approach_target_pos = target_pos.copy()
            self.last_bottle_center_for_forward = look_point.copy()

            # gripper_tcp X축이 물병 중심을 정확히 보게 함
            x_axis = look_point - target_pos
            x_axis = x_axis / np.linalg.norm(x_axis)

            # 그리퍼 벌어지는 Y축은 책상과 평행
            z_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            y_axis = np.cross(z_world, x_axis)

            if np.linalg.norm(y_axis) < 1e-6:
                y_axis = np.array([0.0, 1.0, 0.0])
            else:
                y_axis = y_axis / np.linalg.norm(y_axis)

            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)

            rot_mat = np.column_stack((x_axis, y_axis, z_axis))
            q = Rotation.from_matrix(rot_mat).as_quat()

            target_pose.pose.orientation.x = float(q[0])
            target_pose.pose.orientation.y = float(q[1])
            target_pose.pose.orientation.z = float(q[2])
            target_pose.pose.orientation.w = float(q[3])


            self.target_pose_pub.publish(target_pose)

            self.get_logger().info(
                f"gripper_tcp 목표 Pose 생성 "
                f"x={target_pose.pose.position.x:.3f}, "
                f"y={target_pose.pose.position.y:.3f}, "
                f"z={target_pose.pose.position.z:.3f}"
            )

        except Exception as e:
            self.get_logger().error(f"목표 Pose 계산 실패: {e}")
            self.is_planning = False
            return

        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("/move_action server not available")
            self.is_planning = False
            return

        goal_msg = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = "ur_manipulator"
        request.num_planning_attempts = 1
        request.allowed_planning_time = 1.5
        request.max_velocity_scaling_factor = 0.03
        request.max_acceleration_scaling_factor = 0.03
        request.planner_id = "RRTConnectkConfigDefault"

        # 위치 제약: 이제 tool0 말고 gripper_tcp 기준
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.get_robot_base_frame()
        pos_constraint.link_name = "gripper_tcp"
        pos_constraint.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.08]

        pos_constraint.constraint_region.primitives.append(sphere)
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)

        # 자세 제약: 마지막 목표에서만 gripper_tcp Z축이 물병 축과 일치
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self.get_robot_base_frame()
        ori_constraint.link_name = "gripper_tcp"
        ori_constraint.orientation = target_pose.pose.orientation

        # X/Y는 어느 정도 맞추고, Z축 회전(yaw)은 자유롭게
        if self.bed_mode:
            ori_constraint.absolute_x_axis_tolerance = 1.5
            ori_constraint.absolute_y_axis_tolerance = 1.5
            ori_constraint.absolute_z_axis_tolerance = 3.14
            ori_constraint.weight = 0.2
        else:
            ori_constraint.absolute_x_axis_tolerance = 0.5
            ori_constraint.absolute_y_axis_tolerance = 0.5
            ori_constraint.absolute_z_axis_tolerance = 3.14
            ori_constraint.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(pos_constraint)
        goal_constraints.orientation_constraints.append(ori_constraint)
        if self.current_joint_state is not None:
            for name, pos in zip(self.current_joint_state.name, self.current_joint_state.position):
                if name not in [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint"
                ]:
                    continue

                jc = JointConstraint()
                jc.joint_name = name
                jc.position = float(pos)

                # 기본: 현재 자세 기준 ±90도
                jc.tolerance_above = 1.57
                jc.tolerance_below = 1.57
                jc.weight = 0.2

                # 전체 한바퀴 도는 핵심: shoulder_pan 강하게 제한
                if name == "shoulder_pan_joint":
                    jc.tolerance_above = 1.57
                    jc.tolerance_below = 1.57
                    jc.weight = 0.5


                goal_constraints.joint_constraints.append(jc)

        request.goal_constraints.append(goal_constraints)
        if self.current_joint_state is not None:
            request.start_state = RobotState()
            request.start_state.joint_state = self.current_joint_state
            request.start_state.is_diff = False
        else:
            request.start_state.is_diff = True

        options = PlanningOptions()
        options.plan_only = False
        options.look_around = False
        options.replan = False
        options.replan_attempts = 0
        options.planning_scene_diff.is_diff = True
        request.allowed_planning_time = 1.5
        options.planning_scene_diff.robot_state.is_diff = True

        goal_msg.request = request
        goal_msg.planning_options = options

        self.goal_active = True
        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.move_group_goal_response_callback)   

    def move_side_by_bbox_error(self, error_x):
        if self.current_joint_state is None:
            self.get_logger().warn("X 보정 실패: joint_state 없음")
            return

        joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        pos_dict = dict(zip(
            self.current_joint_state.name,
            self.current_joint_state.position
        ))

        target_pos = [float(pos_dict[name]) for name in joint_order]

        # bbox가 오른쪽이면 shoulder_pan을 아주 조금 회전
        step = 0.006  # rad, 약 0.34도

        if error_x > 0:
            target_pos[0] -= step
        else:
            target_pos[0] += step

        traj = JointTrajectory()
        traj.joint_names = joint_order

        point = JointTrajectoryPoint()
        point.positions = target_pos
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 300_000_000

        traj.points.append(point)
        self.joint_traj_pub.publish(traj)

        self.get_logger().warn(
            f"SCALED X 보정: error_x={error_x}, shoulder_pan step={np.degrees(step):.2f} deg"
        )
        
    def move_xy_by_bbox_error(self, error_x, error_y):
        if self.current_joint_state is None:
            self.get_logger().warn("XY 보정 실패: joint_state 없음")
            return

        joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        pos_dict = dict(zip(
            self.current_joint_state.name,
            self.current_joint_state.position
        ))

        target_pos = [float(pos_dict[name]) for name in joint_order]

        x_step = 0.006
        y_step = 0.004

        # 화면 좌우 보정
        if abs(error_x) >= self.pixel_tolerance:
            if error_x > 0:
                target_pos[0] -= x_step
            else:
                target_pos[0] += x_step

        # 화면 상하 보정
        # 물체가 화면 아래에 있으면 TCP를 아래로 맞추는 방향
        if abs(error_y) >= self.pixel_tolerance_y:
            if error_y > 0:
                target_pos[1] += y_step
            else:
                target_pos[1] -= y_step

        traj = JointTrajectory()
        traj.joint_names = joint_order

        point = JointTrajectoryPoint()
        point.positions = target_pos
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 300_000_000

        traj.points.append(point)
        self.joint_traj_pub.publish(traj)

        self.get_logger().warn(
            f"XY 보정: error_x={error_x}, error_y={error_y}, "
            f"pan_step={np.degrees(x_step):.2f}, "
            f"lift_step={np.degrees(y_step):.2f}"
        )

    def move_group_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("MoveGroup goal rejected")
            self.is_planning = False
            self.goal_active = False
            return
        self.get_logger().info("MoveGroup goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.move_group_result_callback)

    def move_group_result_callback(self, future):
        error_code = None
        try:
            result = future.result().result
            error_code = result.error_code.val
            if error_code == 1:
                self.get_logger().info("MoveGroup motion success")

                if self.latest_bottle_pose_type == "standing":
                    self.get_logger().warn("STANDING BOTTLE -> FLOOR ALIGN")
                    self.align_gripper_y_parallel_to_floor_by_wrist3()
                else:
                    self.get_logger().warn("LYING BOTTLE -> PCA ALIGN")
                    self.align_gripper_y_to_bottle_axis_by_wrist3()
                self.start_tracking_timer = self.create_timer(2.2, self.start_tracking_once)
            else:
                self.get_logger().warn(f"MoveGroup failed. error_code={error_code}")
        except Exception as e:
            self.get_logger().error(f"MoveGroup result error: {e}")
        finally:
            self.is_planning = False

            # 성공이면 wrist 보정 + tracking 시작 전까지 goal_active 유지
            if error_code != 1:
                self.goal_active = False
    def get_robot_base_frame(self):
        return self.base_frame


    def update_gripper_tcp_position(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.gripper_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05)) #self.target_object (bottle).
            self.current_tcp_pos = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z], dtype=np.float64)
        except Exception as e:
            self.get_logger().debug(f'gripper_tcp TF unavailable: {e}')


    def get_latest_cloud(self):
        with self._cloud_lock:
            return self.cloud_msg
            
    def update_locked_target(self, bottle_base):
        p = np.array(bottle_base, dtype=np.float64)

        if self.target_locked:
            return self.locked_bottle_base

        self.target_lock_buffer.append(p)

        if len(self.target_lock_buffer) > self.target_lock_need:
            self.target_lock_buffer.pop(0)

        if len(self.target_lock_buffer) < self.target_lock_need:
            self.get_logger().warn(
                f"물병 좌표 안정화 중: "
                f"{len(self.target_lock_buffer)}/{self.target_lock_need}"
            )
            return None

        pts = np.array(self.target_lock_buffer)
        mean_pt = np.mean(pts, axis=0)

        spread = np.max(np.linalg.norm(pts - mean_pt, axis=1))

        if spread > self.target_lock_spread_limit:
            self.get_logger().warn(
                f"좌표 흔들림 큼: spread={spread*100:.1f}cm "
                f"> {self.target_lock_spread_limit*100:.1f}cm → 다시 수집"
            )
            self.target_lock_buffer.clear()
            return None

        self.locked_bottle_base = mean_pt
        self.target_locked = True
        self.target_ever_locked = True

        self.get_logger().warn(
            f"물병 좌표 LOCK 완료: "
            f"x={mean_pt[0]:.3f}, y={mean_pt[1]:.3f}, z={mean_pt[2]:.3f}"
        )

        return self.locked_bottle_base
        

    def shutdown_once(self):

        if hasattr(self, "shutdown_timer") and self.shutdown_timer is not None:
            self.shutdown_timer.cancel()
            self.shutdown_timer = None

        msg = Bool()
        msg.data = True

        self.grasp_done_pub.publish(msg)

        self.get_logger().warn(
            "작업 완료 → /grasp_done=True publish"
        )

        time.sleep(0.5)

        rclpy.shutdown()

    def read_points_roi_ordered(self, cloud_msg, x1, y1, x2, y2, step=2):
        if cloud_msg is None:
            return None
        h = cloud_msg.height
        w = cloud_msg.width
        if h <= 1:
            return None

        x1, x2 = max(0, min(w - 1, int(x1))), max(0, min(w - 1, int(x2)))
        y1, y2 = max(0, min(h - 1, int(y1))), max(0, min(h - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None

        arr = pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=False)
        arr = np.asarray(arr)

        if arr.dtype.names is not None:
            pts = np.column_stack((arr['x'].reshape(-1), arr['y'].reshape(-1), arr['z'].reshape(-1))).astype(np.float32)
        else:
            pts = arr.astype(np.float32).reshape(-1, 3)

        if pts.shape[0] != h * w:
            return None

        pts_img = pts.reshape(h, w, 3)
        roi = pts_img[y1:y2:step, x1:x2:step, :]
        pts = roi.reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = pts[(pts[:, 2] > 0.2) & (pts[:, 2] < 2.0)]

        if len(pts) < 5:
            return None
        return pts

    def transform_point_to_base(self, cloud_msg, camera_point, target_frame=None):

        source_frame = (
            cloud_msg.header.frame_id.strip('/')
            if cloud_msg is not None
            else 'camera_depth_optical_frame'
        )

        if target_frame is None:
            target_frame = self.get_robot_base_frame()

        if target_frame == source_frame:
            return [
                float(camera_point[0]),
                float(camera_point[1]),
                float(camera_point[2])
            ]

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            pose_src = PoseStamped()
            pose_src.header.frame_id = source_frame
            pose_src.header.stamp = self.get_clock().now().to_msg()

            pose_src.pose.position.x = float(camera_point[0])
            pose_src.pose.position.y = float(camera_point[1])
            pose_src.pose.position.z = float(camera_point[2])
            pose_src.pose.orientation.w = 1.0

            pose_dst = do_transform_pose(
                pose_src.pose,
                transform
            )

            return [
                pose_dst.position.x,
                pose_dst.position.y,
                pose_dst.position.z
            ]

        except Exception as e:
            self.get_logger().warn(
                f"TF transform failed: "
                f"{source_frame} -> {target_frame}, {e}"
            )
            return None

    def read_table_points_ordered(self, cloud_msg, img_w, img_h, step=4):
        return self.read_points_roi_ordered(cloud_msg, 0, img_h // 4, img_w - 1, img_h - 1, step=step)

    def make_o3d_cloud(self, pts):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        return cloud

    def detect_table_plane(self, pts):
        if pts is None or len(pts) < 50:
            return None, None
        cloud = self.make_o3d_cloud(pts)
        cloud = cloud.voxel_down_sample(0.01)
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)

        if len(cloud.points) < 30:
            return None, None

        plane_model, inliers = cloud.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=500)
        a, b, c, d = plane_model
        
        # 평면 노말 부호 정렬 보정
        if c < 0:
            a, b, c, d = -a, -b, -c, -d
            plane_model = (a, b, c, d)

        normal_vector = np.array([a, b, c])
        normal_vector /= np.linalg.norm(normal_vector) 

        cosine_similarity = np.abs(np.dot(normal_vector, np.array([0.0, 0.0, 1.0])))
        angle_deg = np.degrees(np.arccos(np.clip(cosine_similarity, 0.0, 1.0)))

        if angle_deg > 70.0:
            return None, None

        table_pts = np.asarray(cloud.points)[inliers]
        if len(table_pts) < 30:
            return None, None
        return plane_model, table_pts

    def extract_bottle_cluster(self, bottle_pts, plane_model):
        if bottle_pts is None or plane_model is None:
            return None, None

        a, b, c, d = plane_model
        signed_dist = (
            a * bottle_pts[:, 0]
            + b * bottle_pts[:, 1]
            + c * bottle_pts[:, 2]
            + d
        ) / np.sqrt(a*a + b*b + c*c)

        # 책상면에서 4.5cm 이상 위 점만 물체 후보
        obj_pts = bottle_pts[signed_dist > 0.045]
        if len(obj_pts) < 20:
            return None, None

        cloud = self.make_o3d_cloud(obj_pts)
        cloud = cloud.voxel_down_sample(0.008)
        labels = np.array(
            cloud.cluster_dbscan(
                eps=0.025,
                min_points=6,
                print_progress=False
            )
        )

        if labels.size == 0 or labels.max() < 0:
            return None, None

        labels_unique = np.unique(labels)
        labels_unique = labels_unique[labels_unique >= 0]

        cloud_pts = np.asarray(cloud.points)

        roi_center = np.mean(obj_pts, axis=0)

        best_cluster = None
        best_score = 1e9

        for label in labels_unique:
            pts_label = cloud_pts[labels == label]

            if len(pts_label) < 10:
                continue

            center = np.mean(pts_label, axis=0)
            size = np.max(pts_label, axis=0) - np.min(pts_label, axis=0)

            height = np.max(size)

            if height < 0.08:
                continue

            if size[0] > 0.25 or size[1] > 0.25 or size[2] > 0.35:
                continue

            dist_to_roi_center = np.linalg.norm(center - roi_center)

            # 너무 큰 군집은 배경/책상일 가능성이 높아서 벌점
            size_penalty = np.linalg.norm(size) * 0.5

            score = dist_to_roi_center + size_penalty


            if score < best_score:
                best_score = score
                best_cluster = pts_label

        if best_cluster is None:
            return None, None

        cluster_pts = best_cluster
        center = np.mean(cluster_pts, axis=0)

        if len(cluster_pts) < 15:
            return None, None

        centroid = np.mean(cluster_pts, axis=0)
        min_xyz = np.min(cluster_pts, axis=0)
        max_xyz = np.max(cluster_pts, axis=0)
        bbox_size = max_xyz - min_xyz
        bbox_center = (min_xyz + max_xyz) / 2.0
        # PCA로 물체 장축 방향 계산
        # 기존 3D PCA 대신 XY 평면 PCA
        centered_xy = cluster_pts[:, :2] - np.mean(cluster_pts[:, :2], axis=0)
        cov_xy = np.cov(centered_xy.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_xy)

        main_axis_xy = eig_vecs[:, np.argmax(eig_vals)]
        main_axis_xy = main_axis_xy / np.linalg.norm(main_axis_xy)

        main_axis = np.array([
            main_axis_xy[0],
            main_axis_xy[1],
            0.0
        ], dtype=np.float64)

        main_axis = main_axis / np.linalg.norm(main_axis)
        # PCA 축 방향 180도 뒤집힘 방지
        # 항상 현재 TCP에서 물병 쪽을 향하는 방향과 비슷한 부호로 고정
        if self.current_tcp_pos is not None:
            tcp_xy = np.array(self.current_tcp_pos[:2], dtype=np.float64)
            obj_xy = bbox_center[:2]
            tcp_to_obj = obj_xy - tcp_xy

            if np.linalg.norm(tcp_to_obj) > 1e-6:
                tcp_to_obj = tcp_to_obj / np.linalg.norm(tcp_to_obj)

                if np.dot(main_axis[:2], tcp_to_obj) < 0:
                    main_axis = -main_axis

        yaw_axis = np.arctan2(main_axis[1], main_axis[0])

        # 파지 목표점 계산
        grasp_point = np.array([
            bbox_center[0],
            bbox_center[1],
            bbox_center[2]
        ])

        info = {
            'centroid': centroid,
            'grasp_point': grasp_point,
            'main_axis': main_axis,
            'yaw_axis': yaw_axis,
            'min_xyz': min_xyz,
            'max_xyz': max_xyz,
            'size': bbox_size,
            'bbox_center': bbox_center,
        }
        return cluster_pts, info
    
    def lift_then_return_once(self):
        if hasattr(self, "lift_timer") and self.lift_timer is not None:
            self.lift_timer.cancel()
            self.lift_timer = None

        self.get_logger().warn("그리퍼 닫은 후 리프트 시작")
        self.lift_up_5cm()


    def lift_up_5cm(self):
        self.update_gripper_tcp_position()

        if self.current_tcp_pos is None:
            self.get_logger().warn("lift 실패: current_tcp_pos 없음")
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            target_pose = PoseStamped()
            target_pose.header.frame_id = self.base_frame
            target_pose.header.stamp = self.get_clock().now().to_msg()

            target_pose.pose.position.x = float(self.current_tcp_pos[0])
            target_pose.pose.position.y = float(self.current_tcp_pos[1])
            target_pose.pose.position.z = float(self.current_tcp_pos[2] + 0.05)

            target_pose.pose.orientation = tf.transform.rotation

            self.send_lift_cartesian_path(target_pose)

        except Exception as e:
            self.get_logger().warn(f"lift pose 실패: {e}")

    def send_lift_cartesian_path(self, target_pose):
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("compute_cartesian_path service 없음")
            return

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.group_name = "ur_manipulator"
        req.link_name = self.gripper_frame

        if self.current_joint_state is not None:
            req.start_state = RobotState()
            req.start_state.joint_state = self.current_joint_state
            req.start_state.is_diff = False

        req.waypoints.append(target_pose.pose)
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self.lift_cartesian_callback)

    def lift_cartesian_callback(self, future):
        try:
            res = future.result()

            if res.fraction < 0.9:
                self.get_logger().warn(f"lift Cartesian 실패: fraction={res.fraction:.2f}")
                return

            traj = res.solution.joint_trajectory
            traj.header.stamp = self.get_clock().now().to_msg()

            speed_scale = 2.0

            for p in traj.points:
                p.velocities = []
                p.accelerations = []
                p.effort = []

                total_ns = (
                    p.time_from_start.sec * 1_000_000_000
                    + p.time_from_start.nanosec
                )
                total_ns = int(total_ns * speed_scale)
                p.time_from_start.sec = total_ns // 1_000_000_000
                p.time_from_start.nanosec = total_ns % 1_000_000_000

            self.joint_traj_pub.publish(traj)

            last_t = traj.points[-1].time_from_start
            done_time = last_t.sec + last_t.nanosec * 1e-9 + 0.5

            self.return_timer = self.create_timer(
                done_time,
                self.return_to_fixed_pose_once
            )

            self.get_logger().warn("lift Cartesian 실행 → 완료 후 return")

        except Exception as e:
            self.get_logger().warn(f"lift Cartesian error: {e}")

    def return_to_fixed_pose_once(self):
        if hasattr(self, "return_timer") and self.return_timer is not None:
            self.return_timer.cancel()
            self.return_timer = None

        joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        return_pose = [
            4.558644771575928,
            -0.13444264352832036,
            -1.6154195070266724,
            -1.8358494244017542,
            1.6432443857192993,
            -1.6180499235736292
        ]

        traj = JointTrajectory()
        traj.joint_names = joint_order

        point = JointTrajectoryPoint()
        point.positions = return_pose
        point.time_from_start.sec = 5

        traj.points.append(point)
        self.joint_traj_pub.publish(traj)

        self.motion_done = True
        self.advance_done = True

        self.get_logger().warn("Return fixed pose sent → 6초 후 코드 종료")

        self.shutdown_timer = self.create_timer(
            6.0,
            self.shutdown_once
        )

    def stop_robot_soft(self):
        if self.current_joint_state is None:
            self.get_logger().warn("STOP 실패: joint_state 없음")
            return

        joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        pos_dict = dict(zip(
            self.current_joint_state.name,
            self.current_joint_state.position
        ))

        current_pos = [float(pos_dict[name]) for name in joint_order]

        traj = JointTrajectory()
        traj.joint_names = joint_order

        point = JointTrajectoryPoint()
        point.positions = current_pos
        point.time_from_start.sec = 1

        traj.points.append(point)
        self.joint_traj_pub.publish(traj)

        self.goal_active = False
        self.is_planning = False
        self.advancing = False
        self.tracking_mode = False

        self.get_logger().warn("현재 자세 유지 STOP 명령 전송")

    def transform_points_to_base(self, cloud_msg, points_cam):
        points_cam = np.asarray(points_cam)

        # 점 1개 [x, y, z]가 들어온 경우
        if points_cam.ndim == 1 and points_cam.shape[0] == 3:
            return self.transform_point_to_base(cloud_msg, points_cam)

        # 점 여러 개 [[x,y,z], [x,y,z], ...]가 들어온 경우
        points_base = []

        for p in points_cam:
            pb = self.transform_point_to_base(cloud_msg, p)
            if pb is not None:
                points_base.append(pb)

        if len(points_base) == 0:
            return None

        return np.array(points_base, dtype=np.float32)

    def publish_pc_to_rviz(self, publisher, pts, original_msg):
        if pts is None or original_msg is None:
            return
        from std_msgs.msg import Header
        header = Header()
        header.frame_id = self.get_robot_base_frame()
        header.stamp = self.get_clock().now().to_msg()
        
        pc_msg = pc2.create_cloud_xyz32(header, pts.astype(np.float32))
        publisher.publish(pc_msg)


    def set_io(self, pin, state):
        req = SetIO.Request()
        req.fun = 1
        req.pin = pin
        req.state = float(state)

        self.io_client.call_async(req)

    def gripper_open(self):
        self.set_io(17, 0.0)
        self.set_io(16, 1.0)

    def gripper_close(self):
        self.set_io(16, 0.0)
        self.set_io(17, 1.0)
        self.get_logger().info("Gripper CLOSE")

    def send_forward_to_bottle_15cm(self):
        self.update_gripper_tcp_position()

        if self.latest_bottle_base is None:
            self.get_logger().warn("전진 실패: latest_bottle_base 없음")
            return

        if self.current_tcp_pos is None:
            self.get_logger().warn("전진 실패: current_tcp_pos 없음")
            return

        if self.advancing:
            return

        self.advancing = True

        try:
            p_tcp = np.array(self.current_tcp_pos, dtype=np.float64)
            # 현재 gripper_tcp 자세 읽기
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            q = tf.transform.rotation
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
            R = rot.as_matrix()

            if not hasattr(self, "last_approach_target_pos") or not hasattr(self, "last_bottle_center_for_forward"):
                self.get_logger().warn("전진 실패: 저장된 target_pose 또는 bottle_center 없음")
                self.advancing = False
                return

            # gripper_tcp X축 방향으로 직진
            p_bottle = np.array(self.last_bottle_center_for_forward, dtype=np.float64)

            # 거리 숫자만 target_pose 기준으로 계산
            # 거리 숫자만 target_pose 기준으로 계산
            direction = R[:, 0]
            direction = direction / np.linalg.norm(direction)

            # X 보정 끝난 현재 TCP 위치 기준으로 물병까지 남은 거리 계산
            raw_forward_dist = np.dot(
                (p_bottle - p_tcp),
                direction
            )
            dist_to_bottle = raw_forward_dist

            grasp_margin = 0.00
            max_forward = 0.16

            forward_dist = max(0.0, raw_forward_dist - grasp_margin)
            forward_dist = min(forward_dist, max_forward)

            target_pos = p_tcp + direction * forward_dist

            self.get_logger().warn(
                f"X보정 후 전진거리 재계산: "
                f"raw={raw_forward_dist*100:.1f}cm, "
                f"actual={forward_dist*100:.1f}cm, "
                f"tcp=({p_tcp[0]:.3f},{p_tcp[1]:.3f},{p_tcp[2]:.3f}), "
                f"bottle=({p_bottle[0]:.3f},{p_bottle[1]:.3f},{p_bottle[2]:.3f})"
            )

            self.get_logger().warn(
                f"전진거리 자동계산: target~bottle={dist_to_bottle*100:.1f}cm, "
                f"실제전진={forward_dist*100:.1f}cm, "
                f"gripper_x_dir=({direction[0]:.3f},{direction[1]:.3f},{direction[2]:.3f})"
            )


            target_pose = PoseStamped()
            target_pose.header.frame_id = self.base_frame
            target_pose.header.stamp = self.get_clock().now().to_msg()

            target_pose.pose.position.x = float(target_pos[0])
            target_pose.pose.position.y = float(target_pos[1])
            target_pose.pose.position.z = float(target_pos[2])

            target_pose.pose.orientation = tf.transform.rotation

            self.get_logger().warn(
                f"FORWARD 15cm 목표: "
                f"x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}"
            )

            self.send_forward_cartesian_path(target_pose)

        except Exception as e:
            self.get_logger().warn(f"15cm 전진 실패: {e}")
            self.advancing = False

    def publish_bottle_marker(self, bottle_base, size):
        marker = Marker()
        marker.header.frame_id = self.get_robot_base_frame()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detected_bottle"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = bottle_base[0]
        marker.pose.position.y = bottle_base[1]
        marker.pose.position.z = bottle_base[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = float(size[0])
        marker.scale.y = float(size[1])
        marker.scale.z = float(size[2])
        
        marker.color.r = 0.2
        marker.color.g = 0.9
        marker.color.b = 0.2
        marker.color.a = 0.65
        self.marker_pub.publish(marker)

        text_marker = Marker()
        text_marker.header = marker.header
        text_marker.ns = "bottle_label"
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = bottle_base[0]
        text_marker.pose.position.y = bottle_base[1]
        text_marker.pose.position.z = bottle_base[2] + float(size[2])/2.0 + 0.05
        text_marker.scale.z = 0.04
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"Target: {self.target_object.upper()}"
        self.marker_pub.publish(text_marker)

    def search_z_motion(self):
        if time.time() - self.node_start_time < self.z_search_start_delay:
            self.get_logger().warn("초기 안정화 중: Z SEARCH 차단")
            return

        if self.advancing or self.is_planning or self.goal_active:
            return
        if self.search_z_busy:
            return

        now = time.time()

        if hasattr(self, "last_search_z_time"):
            if now - self.last_search_z_time < 3.0:
                return

        self.search_z_busy = True
        self.last_search_z_time = now

        self.update_gripper_tcp_position()

        if self.current_tcp_pos is None:
            self.get_logger().warn("Z SEARCH 실패: current_tcp_pos 없음")
            self.search_z_busy = False
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            target_pose = PoseStamped()
            target_pose.header.frame_id = self.base_frame
            target_pose.header.stamp = self.get_clock().now().to_msg()
            target_pose.pose.orientation = tf.transform.rotation

            dz = self.search_z_step * self.search_z_dir

            target_pose.pose.position.x = float(self.current_tcp_pos[0])
            target_pose.pose.position.y = float(self.current_tcp_pos[1])
            target_pose.pose.position.z = float(self.current_tcp_pos[2] + dz)

            self.get_logger().warn(
                f"YOLO Z SEARCH Cartesian: dz={dz*100:.1f}cm"
            )

            self.send_search_z_cartesian_path(target_pose)

        except Exception as e:
            self.get_logger().warn(f"Z SEARCH 실패: {e}")
            self.search_z_busy = False

    def send_search_z_cartesian_path(self, target_pose):
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("compute_cartesian_path service 없음")
            self.search_z_busy = False
            return

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.group_name = "ur_manipulator"
        req.link_name = self.gripper_frame

        if self.current_joint_state is not None:
            req.start_state = RobotState()
            req.start_state.joint_state = self.current_joint_state
            req.start_state.is_diff = False

        req.waypoints.append(target_pose.pose)
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self.search_z_cartesian_callback)


    def search_z_cartesian_callback(self, future):
        try:
            res = future.result()

            if res.fraction < 0.8:
                self.get_logger().warn(f"Z SEARCH Cartesian 실패: fraction={res.fraction:.2f}")
                self.search_z_busy = False
                return

            traj = res.solution.joint_trajectory
            traj.header.stamp = self.get_clock().now().to_msg()

            speed_scale = 10.0

            for p in traj.points:
                p.velocities = []
                p.accelerations = []
                p.effort = []

                total_ns = (
                    p.time_from_start.sec * 1_000_000_000
                    + p.time_from_start.nanosec
                )
                total_ns = int(total_ns * speed_scale)
                p.time_from_start.sec = total_ns // 1_000_000_000
                p.time_from_start.nanosec = total_ns % 1_000_000_000

            self.joint_traj_pub.publish(traj)

            last_t = traj.points[-1].time_from_start
            move_time = last_t.sec + last_t.nanosec * 1e-9

            self.get_logger().warn("Z SEARCH Cartesian 실행 → 이동 후 2초 대기")

            self.search_z_timer = self.create_timer(
                move_time + self.search_z_wait_sec,
                self.finish_search_z_once
            )

        except Exception as e:
            self.get_logger().warn(f"Z SEARCH Cartesian error: {e}")

    def finish_search_z_once(self):
        if hasattr(self, "search_z_timer") and self.search_z_timer is not None:
            self.search_z_timer.cancel()
            self.search_z_timer = None

        self.search_z_dir *= -1.0
        self.search_z_busy = False

        self.get_logger().warn("Z SEARCH 대기 완료 → 다음 방향 준비")

    def show_debug_image(self, img):
        
        if not self.show_visualization:
            return

        img_small = cv2.resize(img, (960, 540))
        cv2.imshow("PC RANSAC Test", img_small)
        cv2.waitKey(1)

    def color_callback(self, msg):
        if self.waiting_start:
            return
        
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return
        
        current_cloud = self.get_latest_cloud()

        
        # 로봇 움직이는 동안은 카메라 화면만 보여주고
        # PointCloud 계산은 하지 않음
        # 로봇 움직이는 동안은 카메라 화면 끄고 계산도 안 함
        if self.goal_active or self.is_planning or self.advancing:
            cv2.putText(
                img,
                "BUSY: motion/planning paused detection",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            self.show_debug_image(img)
            return

        h, w = img.shape[:2]
        results = self.model(img, verbose=False)
        found = False
        x1 = y1 = x2 = y2 = 0

        for r in results:
            for box in r.boxes:
                name = self.model.names[int(box.cls[0])]
                if name == self.target_object:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    found = True
                    break
            if found:
                break

        if not found:
            if self.target_ever_locked:
                cv2.putText(
                    img,
                    "Bottle not found - target already locked",
                    (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                self.show_debug_image(img)
                return

            cv2.putText(
                img,
                "Bottle not found - Z SEARCHING",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                2
            )

            self.latest_bottle_base = None
            self.search_z_motion()
            self.show_debug_image(img)
            return

        cx_color = int((x1 + x2) / 2)
        cy_color = int((y1 + y2) / 2)

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_h > bbox_w * 1.3:
            self.latest_bottle_pose_type = "standing"
        else:
            self.latest_bottle_pose_type = "lying"

        self.get_logger().warn(
            f"BOTTLE TYPE = {self.latest_bottle_pose_type}"
        )

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )



        stop = TwistStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.header.frame_id = self.base_frame
        self.servo_pub.publish(stop)

        # 도착 후 X 보정 모드
        if self.tracking_mode and not self.tracking_done:
            image_center_x = w // 2 + self.camera_x_offset_px
            image_center_y = h // 2 + self.camera_y_offset_px

            error_x = cx_color - image_center_x
            error_y = cy_color - image_center_y

            cv2.line(
                img,
                (image_center_x, h // 2),
                (cx_color, cy_color),
                (0, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"X TRACK error={error_x}",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            self.get_logger().warn(
                f"YOLO X TRACK: bbox_cx={cx_color}, img_cx={image_center_x}, error_x={error_x}"
            )

            if abs(error_x) < self.pixel_tolerance and abs(error_y) < self.pixel_tolerance_y:
                stop = TwistStamped()
                stop.header.stamp = self.get_clock().now().to_msg()
                stop.header.frame_id = self.get_robot_base_frame()
                self.servo_pub.publish(stop)

                self.tracking_done = True
                self.tracking_mode = False

                self.get_logger().warn("X 보정 완료 → 15cm 전진")
                self.send_forward_to_bottle_15cm()
                self.show_debug_image(img)
                return

            self.move_xy_by_bbox_error(error_x, error_y)
            self.show_debug_image(img)
            return

        # YOLO는 잡혔지만 PointCloud가 없으면 계속 Z SEARCH
        if current_cloud is None:
            cv2.putText(
                img,
                "Waiting PointCloud... Z SEARCHING",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            self.search_z_motion()
            self.show_debug_image(img)
            return

        table_pts = self.read_table_points_ordered(current_cloud, w, h, step=4)
        plane_model, table_inliers = self.detect_table_plane(table_pts)
        
        if table_inliers is not None:
            table_inliers_base_viz = self.transform_points_to_base(
                current_cloud,
                table_inliers
            )

            if table_inliers_base_viz is not None:
                self.publish_pc_to_rviz(
                    self.table_pc_pub,
                    table_inliers_base_viz,
                    current_cloud
                )

        sx = current_cloud.width / w
        sy = current_cloud.height / h


        # pointcloud 좌표계 기준 중심

        sx = current_cloud.width / w
        sy = current_cloud.height / h

        x1_cloud = int(x1 * sx)
        y1_cloud = int(y1 * sy)
        x2_cloud = int(x2 * sx)
        y2_cloud = int(y2 * sy)

        self.get_logger().warn("PointCloud ROI: YOLO box 사용")



        roi_scale = 0.6

        cx = (x1_cloud + x2_cloud) // 2
        cy = (y1_cloud + y2_cloud) // 2

        half_w = int((x2_cloud - x1_cloud) * roi_scale * 0.5)
        half_h = int((y2_cloud - y1_cloud) * roi_scale * 0.5)

        rx1 = cx - half_w
        rx2 = cx + half_w
        ry1 = cy - half_h
        ry2 = cy + half_h

        bottle_pts = self.read_points_roi_ordered(
            current_cloud,
            rx1,
            ry1,
            rx2,
            ry2,
            step=1
        )



        if bottle_pts is None or len(bottle_pts) < 10:
            cv2.putText(
                img,
                "CENTER DEPTH NONE -> Z SEARCH",
                (40, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            self.search_z_motion()
            self.show_debug_image(img)
            return

        # 중앙 작은 ROI의 median 값을 물병 목표점으로 사용
        # 투명 물병은 median이 뒤 배경으로 튀므로, 가까운 depth 쪽 점을 사용
        z_vals = bottle_pts[:, 2]

        z_near = np.percentile(z_vals, 15)

        near_pts = bottle_pts[
            np.abs(bottle_pts[:, 2] - z_near) < 0.03
        ]

        if len(near_pts) < 10:
            z_near = np.min(z_vals)
            near_pts = bottle_pts[
                np.abs(bottle_pts[:, 2] - z_near) < 0.05
            ]

        if len(near_pts) < 10:
            near_pts = bottle_pts

        #centroid_cam = np.median(near_pts, axis=0)
        if self.camera_info is None:
            self.get_logger().warn("Waiting camera_info...")
            self.show_debug_image(img)
            return


        centroid_cam = np.median(
            near_pts,
            axis=0
        )


        
        cluster_pts, bottle_info = self.extract_bottle_cluster(
            bottle_pts,
            plane_model
        )

        if bottle_info is None:
            self.get_logger().warn("Bottle cluster extraction failed. Use near_pts fallback.")

            bottle_info = {
                'centroid': centroid_cam,
                'grasp_point': centroid_cam,
                'main_axis': np.array([1.0, 0.0, 0.0]),
                'yaw_axis': 0.0,
                'size': np.array([0.07, 0.07, 0.20])
            }

            cluster_pts = near_pts

        # if bottle_info is not None:
        #     self.get_logger().info(
        #         f"물병 PCA yaw: "
        #         f"{np.degrees(bottle_info['yaw_axis']):.2f} deg"
        #     )
        
        main_axis_base = self.transform_vector_to_base(
            current_cloud,
            bottle_info['main_axis']
        )

        if main_axis_base is not None:
            self.latest_main_axis_base = main_axis_base
            self.get_logger().warn(
                f"PCA AXIS 사용: "
                f"base_axis=({main_axis_base[0]:.2f},"
                f"{main_axis_base[1]:.2f},"
                f"{main_axis_base[2]:.2f})"
            )
        else:
            self.latest_main_axis_base = None
            self.get_logger().warn("PCA AXIS 변환 실패")
        if cluster_pts is not None:
            cluster_pts_base = self.transform_points_to_base(
                current_cloud,
                cluster_pts
            )

            if cluster_pts_base is not None:
                self.publish_pc_to_rviz(
                    self.bottle_pc_pub,
                    cluster_pts_base,
                    current_cloud
                )

        if bottle_info is not None:
            centroid_cam = bottle_info['grasp_point'].copy()
            # surface_point_cam = bottle_info['grasp_point'].copy()

            # object_radius = 0.035  # 물병 반지름

            # view_dir = surface_point_cam / np.linalg.norm(surface_point_cam)

            # center_point_cam = (
            #     surface_point_cam
            #     + view_dir * object_radius
            # )

            # centroid_cam = center_point_cam
            
            # offset_x = -0.05  
            # offset_y = 0.00
            
            # centroid_cam[0] -= offset_x
            # centroid_cam[1] -= offset_y

            self.get_logger().info(f"\n[STEP 1] 변환 전 (Camera Frame): X={centroid_cam[0]:.4f}, Y={centroid_cam[1]:.4f}, Z(실거리)={centroid_cam[2]:.4f}")
            # [수정] 모서리 한 점 대신, 테이블 전체 인라이어 포인트의 평균 중심점(Centroid) 계산
            #table_centroid_cam = np.mean(table_inliers, axis=0)


            bottle_base = self.transform_point_to_base(
                current_cloud,
                centroid_cam
            )
            
            if table_inliers is not None and not self.bed_mode:
                table_base = self.transform_points_to_base(current_cloud, table_inliers)

                if table_base is not None:
                    table_z = np.median(table_base[:, 2])
                    min_target_z = table_z + 0.06

                    if bottle_base[2] < min_target_z:
                        self.get_logger().warn(
                            f"target z too low -> clamp: "
                            f"{bottle_base[2]:.3f} -> {min_target_z:.3f}"
                        )
                        bottle_base[2] = float(min_target_z)

            if bottle_base is None:
                self.get_logger().warn("bottle_base transform failed. Skip this frame.")
                return

            self.get_logger().info(
                f"target base = "
                f"{bottle_base[0]:.3f}, "
                f"{bottle_base[1]:.3f}, "
                f"{bottle_base[2]:.3f}"
            )
            target_marker = Marker()
            target_marker.header.frame_id = self.get_robot_base_frame()
            target_marker.header.stamp = self.get_clock().now().to_msg()
            target_marker.ns = "bottle_target_point"
            target_marker.id = 100
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD

            target_marker.pose.position.x = float(bottle_base[0])
            target_marker.pose.position.y = float(bottle_base[1])
            target_marker.pose.position.z = float(bottle_base[2])
            target_marker.pose.orientation.w = 1.0

            target_marker.scale.x = 0.05
            target_marker.scale.y = 0.05
            target_marker.scale.z = 0.05

            target_marker.color.r = 1.0
            target_marker.color.g = 0.0
            target_marker.color.b = 0.0
            target_marker.color.a = 1.0

            self.marker_pub.publish(target_marker)

            # [수정] 테이블 모서리 대신 진짜 '테이블 중심점'을 base_link 좌표계로 변환
            
            if bottle_base is not None:
                self.get_logger().info(f"[STEP 2] 변환 후 (Robot Base Frame): X={bottle_base[0]:.4f}, Y={bottle_base[1]:.4f}, Z(높이)={bottle_base[2]:.4f}\n")
                if abs(bottle_base[0]) > 0.8 or abs(bottle_base[1]) > 0.8 or bottle_base[2] < -0.25 or bottle_base[2] > 1.0:
                    self.get_logger().warn(f"Rejected unsafe target: {bottle_base}")
                    return

                locked_target = self.update_locked_target(bottle_base)

                if locked_target is None:
                    self.show_debug_image(img)
                    return

                self.latest_bottle_base = locked_target.tolist()
                self.latest_bottle_size = bottle_info['size']

                self.publish_bottle_marker(
                    self.latest_bottle_base,
                    self.latest_bottle_size
                )

                # [수정] 정렬된 진짜 테이블 중심 베이스 좌표를 매개변수로 전달
                if table_inliers is not None and not self.bed_mode:
                    table_inliers_base = self.transform_points_to_base(
                        current_cloud,
                        table_inliers
                    )

                    if table_inliers_base is not None and not self.scene_initialized:
                        self.update_planning_scene_cluster_obstacles(
                            table_inliers_base,
                            bottle_base
                        )

                        self.scene_initialized = True

                        self.get_logger().warn("테이블 장애물 등록 완료")

                cv2.putText(img, f"BASE X:{bottle_base[0]:.2f} Y:{bottle_base[1]:.2f} Z:{bottle_base[2]:.2f}", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


            centroid = bottle_info['centroid']
            size = bottle_info['size']
            cv2.putText(img, f"BOTTLE xyz=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f})", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"SIZE=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f})m", (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img, "BOTTLE cluster: None", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        self.show_debug_image(img)

    def robot_control_loop(self):
        if self.waiting_start:
            return
        
        if self.tracking_mode or self.advancing:
            return
        self.get_logger().warn(
            f"CONTROL LOOP: motion_done={self.motion_done}, "
            f"is_planning={self.is_planning}, "
            f"goal_active={self.goal_active}, "
            f"target={self.latest_bottle_base is not None}"
        )

        if self.tracking_mode:
            return

        if self.motion_done:
            return

        if self.latest_bottle_base is None:
            return

        if self.is_planning or self.goal_active:
            return

        now = time.time()
        if now - self.last_plan_time < self.plan_interval:
            return

        target_point = np.array(self.latest_bottle_base, dtype=np.float64).copy()

        self.send_move_group_goal(target_point)

PROCESSES = []
LOCK_FILE = "/tmp/ur3_auto_run.lock"


def run_cmd(cmd, name, wait=0):
    print(f"== {name} ==")
    p = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        preexec_fn=os.setsid
    )
    PROCESSES.append(p)
    if wait > 0:
        time.sleep(wait)
    return p


def cleanup_processes():
    print("== Shutdown subprocesses ==")

    for p in reversed(PROCESSES):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass

    time.sleep(1)

    for p in reversed(PROCESSES):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass

    subprocess.run(r'''
    pkill -9 -f "ur_control.launch.py|ur_robot_driver|dashboard_client|urscript_interface|controller_stopper_node|trajectory_until_node|ros2_control_node|robot_state_helper|move_group|moveit|robot_state_publisher|realsense2_camera|component_container|rs_launch.py|rviz2" || true

    ros2 daemon stop || true
    pkill -9 -f "_ros2_daemon" || true

    rm -f /dev/shm/fastrtps_port* || true
    rm -f /dev/shm/sem.fastrtps_port* || true
    rm -f /tmp/ur3_auto_run.lock || true

    sleep 2
    ros2 daemon start || true
    ''', shell=True, executable="/bin/bash", check=False)
        
ROBOT_IP = "192.168.1.101"
DASHBOARD_PORT = 29999

def send_dashboard_command(command):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((ROBOT_IP, DASHBOARD_PORT))
        s.recv(1024)
        s.sendall(f"{command}\n".encode("utf-8"))
        response = s.recv(1024).decode("utf-8").strip()
        s.close()
        print(f"[Dashboard] {command} -> {response}")
        return response
    except Exception as e:
        print(f"[Dashboard ERROR] {command}: {e}")
        return None


def start_ros_system():
    ROS_SETUP = "/opt/ros/jazzy/setup.bash"
    WS_SETUP = "/home/dongbin/vs_ws/install/setup.bash"
    URDF = "/home/dongbin/vs_ws/src/ur3_control/urdf/ur3_robot.urdf.xacro"

    atexit.register(cleanup_processes)


    subprocess.run(r'''
    echo "== HARD CLEAN OLD ROS/UR PROCESSES =="

    pkill -9 -f "ur_control.launch.py|ur_robot_driver|dashboard_client|urscript_interface|controller_stopper_node|trajectory_until_node|ros2_control_node|robot_state_helper|move_group|moveit|robot_state_publisher|realsense2_camera|component_container|rs_launch.py|rviz2" || true

    ros2 daemon stop || true
    pkill -9 -f "_ros2_daemon" || true

    rm -f /dev/shm/fastrtps_port* || true
    rm -f /dev/shm/sem.fastrtps_port* || true
    rm -f /tmp/ur3_auto_run.lock || true

    sleep 2
    ros2 daemon start || true
    sleep 2
    ''', shell=True, executable="/bin/bash", check=False)
    #send_dashboard_command("stop")

    run_cmd(f"""
    source {ROS_SETUP}
    source {WS_SETUP}
    ros2 launch realsense2_camera rs_launch.py \
      align_depth.enable:=true \
      pointcloud.enable:=true \
      pointcloud.ordered_pc:=true \
      publish_tf:=false 
    """, "1. Realsense launch", 5)
  

    run_cmd(f"""
    source {ROS_SETUP}
    source {WS_SETUP}
    ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur3 \
    robot_ip:=192.168.1.101 \
    reverse_ip:=192.168.1.102 \
    launch_rviz:=false
    """, "2. UR3 driver launch", 2)

    send_dashboard_command("load External_Control.urp")
    time.sleep(1)
    send_dashboard_command("play")
    time.sleep(5)

    #input("티치펜던트에서 External Control → Play 누른 뒤 Enter 누르세요...")

    run_cmd(f"""
    source {ROS_SETUP}
    source {WS_SETUP}
    ros2 run robot_state_publisher robot_state_publisher \
      --ros-args \
      -p robot_description:="$(ros2 run xacro xacro {URDF})"
    """, "3. Custom Robot State Publisher", 5)

    run_cmd(f"""
    source {ROS_SETUP}
    source {WS_SETUP}
    ros2 launch ur3_moveit_config_custom move_group.launch.py
    """, "4. MoveIt move_group launch", 8)

    print("== 5~7. Controller + Initial Pose ==")
    cmd = f"""
    source {ROS_SETUP}
    source {WS_SETUP}

    ros2 control switch_controllers \
    --deactivate forward_position_controller \
    --deactivate forward_velocity_controller \
    --activate scaled_joint_trajectory_controller || true

    ros2 topic echo /joint_states --once >/dev/null && echo "joint_states OK"

    ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller || true
    sleep 1
    ros2 control switch_controllers --activate scaled_joint_trajectory_controller
    sleep 3
    """

    subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        check=False
    )

def main(args=None):
    config = CONFIG["table"]["standing"]["bottle"].copy()

    if os.path.exists(LOCK_FILE):
        print("이미 코드가 실행 중일 가능성이 있음.")
        print(f"강제 실행하려면: rm -f {LOCK_FILE}")
        return

    open(LOCK_FILE, "w").write(str(os.getpid()))

    node = None

    try:
        start_ros_system()

        rclpy.init(args=args)
        node = PointCloudRansacTest(config)

        print("대기 중: /grasp_command + /approach_done 둘 다 받아야 시작")
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Node error: {exc}")
        raise

    finally:
        if node is not None:
            try:
                if rclpy.ok():
                    node.stop_robot_soft()
                    end_time = time.time() + 1.5
                    while time.time() < end_time and rclpy.ok():
                        rclpy.spin_once(node, timeout_sec=0.1)
            except Exception as e:
                print(f"soft stop failed: {e}")

            node.destroy_node()

        cv2.destroyAllWindows()
        cleanup_processes()

        if rclpy.ok():
            rclpy.shutdown()

        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass

if __name__ == '__main__':
    main()

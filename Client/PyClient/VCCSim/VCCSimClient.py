import grpc
import math
from typing import List, Tuple

import numpy as np
from PIL import Image

from . import VCCSim_pb2
from . import VCCSim_pb2_grpc


class VCCSimClient:
    """Client for interacting with VCCSim services."""

    def __init__(self, host: str = "localhost", port: int = 50996, max_message_length: int = 20 * 1024 * 1024):
        """Initialize the VCCSim client."""
        options = [
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length),
        ]

        self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)

        self.record_service = VCCSim_pb2_grpc.RecordingServiceStub(self.channel)
        self.lidar_service = VCCSim_pb2_grpc.LidarServiceStub(self.channel)
        self.camera_service = VCCSim_pb2_grpc.CameraServiceStub(self.channel)
        self.drone_service = VCCSim_pb2_grpc.DroneServiceStub(self.channel)
        self.car_service = VCCSim_pb2_grpc.CarServiceStub(self.channel)
        self.flash_service = VCCSim_pb2_grpc.FlashServiceStub(self.channel)
        self.mesh_service = VCCSim_pb2_grpc.MeshServiceStub(self.channel)
        self.point_cloud_service = VCCSim_pb2_grpc.PointCloudServiceStub(self.channel)
        self.safe_check_service = VCCSim_pb2_grpc.SafeCheckServiceStub(self.channel)

    def close(self) -> None:
        """Close the gRPC channel."""
        self.channel.close()

    # Helper methods
    def _create_robot_name(self, name: str) -> VCCSim_pb2.RobotName:
        return VCCSim_pb2.RobotName(name=name)

    def _create_vec3(self, x: float, y: float, z: float) -> VCCSim_pb2.Vec3f:
        return VCCSim_pb2.Vec3f(x=x, y=y, z=z)

    def _create_indexed_camera(self, robot_name: str, index: int) -> VCCSim_pb2.IndexedCamera:
        return VCCSim_pb2.IndexedCamera(robot_name=self._create_robot_name(robot_name), index=index)

    def _rotation_from_euler(self, roll: float, pitch: float, yaw: float) -> VCCSim_pb2.Rotation:
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return VCCSim_pb2.Rotation(x=x, y=y, z=z, w=w)

    def _create_pose(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> VCCSim_pb2.Pose:
        return VCCSim_pb2.Pose(
            position=self._create_vec3(x, y, z),
            # convert from rpy -> euler
            rotation=self._rotation_from_euler(roll, pitch, yaw),
        )

    def _create_pose_only_yaw(self, x: float, y: float, z: float, yaw: float) -> VCCSim_pb2.PoseYawOnly:
        return VCCSim_pb2.PoseYawOnly(position=self._create_vec3(x, y, z), yaw=yaw)

    # Record Service Methods
    def toggle_recording(self) -> bool:
        response = self.record_service.Recording(VCCSim_pb2.EmptyRequest())
        return response.status

    # LiDAR Service Methods
    def get_lidar_data(self, robot_name: str) -> VCCSim_pb2.LidarData:
        """Fetch raw LiDAR point cloud data for the given robot."""
        request = self._create_robot_name(robot_name)
        return self.lidar_service.GetLiDARData(request)

    def get_lidar_odom(self, robot_name: str) -> VCCSim_pb2.Odometry:
        """Fetch odometry reported by the LiDAR rig."""
        request = self._create_robot_name(robot_name)
        return self.lidar_service.GetLiDAROdom(request)

    def get_lidar_data_and_odom(self, robot_name: str) -> VCCSim_pb2.LidarDataAndOdom:
        """Fetch LiDAR point cloud and odometry in a single call."""
        request = self._create_robot_name(robot_name)
        return self.lidar_service.GetLiDARDataAndOdom(request)

    # Camera Service Methods
    def get_depth_data(self, robot_name: str, index: int) -> VCCSim_pb2.DepthData:
        request = self._create_indexed_camera(robot_name, index)
        return self.camera_service.GetDepthData(request)

    def get_depth_camera_image_size(self, robot_name: str, index: int) -> Tuple[int, int]:
        depth_data = self.get_depth_data(robot_name, index)
        return depth_data.width, depth_data.height

    def get_depth_camera_image_data(self, robot_name: str, index: int) -> List[float]:
        depth_data = self.get_depth_data(robot_name, index)
        return list(depth_data.data)

    def get_depth_camera_point_data(self, robot_name: str, index: int) -> List[Tuple[float, float, float]]:
        request = self._create_indexed_camera(robot_name, index)
        response = self.camera_service.GetDepthPointCloud(request)
        return [(point.x, point.y, point.z) for point in response.data.data]

    def get_rgb_data(self, robot_name: str, index: int) -> VCCSim_pb2.RGBData:
        request = self._create_indexed_camera(robot_name, index)
        return self.camera_service.GetRGBData(request)

    def get_rgb_indexed_camera_image_size(self, robot_name: str, index: int) -> Tuple[int, int]:
        rgb_data = self.get_rgb_data(robot_name, index)
        return rgb_data.width, rgb_data.height

    def get_rgb_indexed_camera_image_data(self, robot_name: str, index: int) -> VCCSim_pb2.RGBData:
        return self.get_rgb_data(robot_name, index)

    def get_seg_data(self, robot_name: str, index: int) -> VCCSim_pb2.RGBData:
        request = self._create_indexed_camera(robot_name, index)
        return self.camera_service.GetSegData(request)

    def get_normal_data(self, robot_name: str, index: int) -> VCCSim_pb2.NormalData:
        request = self._create_indexed_camera(robot_name, index)
        return self.camera_service.GetNormalData(request)

    # Drone Service Methods
    def get_drone_pose(self, robot_name: str) -> VCCSim_pb2.Pose:
        request = self._create_robot_name(robot_name)
        return self.drone_service.GetDronePose(request)

    def send_drone_pose(self, name: str, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> bool:
        pose = self._create_pose(x, y, z, roll, pitch, yaw)
        request = VCCSim_pb2.DronePose(name=name, pose=pose)
        response = self.drone_service.SendDronePose(request)
        return response.status

    def send_drone_path(self, name: str, poses: List[Tuple[float, float, float, float, float, float]]) -> bool:
        path = [self._create_pose(*pose) for pose in poses]
        request = VCCSim_pb2.DronePath(name=name, path=path)
        response = self.drone_service.SendDronePath(request)
        return response.status

    def move_drone_to_next(self, robot_name: str) -> bool:
        request = self._create_robot_name(robot_name)
        response = self.drone_service.MoveToNext(request)
        return response.status

    # Car Service Methods
    def get_car_odom(self, robot_name: str) -> VCCSim_pb2.Odometry:
        request = self._create_robot_name(robot_name)
        return self.car_service.GetCarOdom(request)

    def send_car_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> bool:
        pose = self._create_pose_only_yaw(x, y, z, yaw)
        request = VCCSim_pb2.CarPose(name=name, pose=pose)
        response = self.car_service.SendCarPose(request)
        return response.status

    def send_car_path(self, name: str, poses: List[Tuple[float, float, float, float]]) -> bool:
        path = [self._create_pose_only_yaw(*pose) for pose in poses]
        request = VCCSim_pb2.CarPath(name=name, path=path)
        response = self.car_service.SendCarPath(request)
        return response.status

    def move_car_to_next(self, robot_name: str) -> bool:
        request = self._create_robot_name(robot_name)
        response = self.car_service.MoveToNext(request)
        return response.status

    # Flash Service Methods
    def get_flash_pose(self, robot_name: str) -> VCCSim_pb2.Pose:
        request = self._create_robot_name(robot_name)
        return self.flash_service.GetFlashPose(request)

    def send_flash_pose(self, name: str, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> bool:
        pose = self._create_pose(x, y, z, roll, pitch, yaw)
        request = VCCSim_pb2.DronePose(name=name, pose=pose)
        response = self.flash_service.SendFlashPose(request)
        return response.status

    def send_flash_path(self, name: str, poses: List[Tuple[float, float, float, float, float, float]]) -> bool:
        path = [self._create_pose(*pose) for pose in poses]
        request = VCCSim_pb2.DronePath(name=name, path=path)
        response = self.flash_service.SendFlashPath(request)
        return response.status

    def check_flash_ready(self, robot_name: str) -> bool:
        request = self._create_robot_name(robot_name)
        response = self.flash_service.CheckReady(request)
        return response.status

    def move_flash_to_next(self, robot_name: str) -> bool:
        request = self._create_robot_name(robot_name)
        response = self.flash_service.MoveToNext(request)
        return response.status

    def move_to_next(self, robot_name: str) -> bool:
        """Backward compatible alias for move_flash_to_next."""
        return self.move_flash_to_next(robot_name)

    # Mesh Service Methods
    def send_mesh(self, data: bytes, format: int, version: int, simplified: bool, transform_pose: Tuple[float, float, float, float, float, float]) -> bool:
        transform = self._create_pose(*transform_pose)
        request = VCCSim_pb2.MeshData(
            data=data,
            format=format,
            version=version,
            simplified=simplified,
            transform=transform,
        )
        response = self.mesh_service.SendMesh(request)
        return response.status

    def send_global_mesh(self, data: bytes, format: int, version: int, simplified: bool, transform_pose: Tuple[float, float, float, float, float, float]) -> int:
        transform = self._create_pose(*transform_pose)
        request = VCCSim_pb2.MeshData(
            data=data,
            format=format,
            version=version,
            simplified=simplified,
            transform=transform,
        )
        response = self.mesh_service.SendGlobalMesh(request)
        return response.id

    def remove_global_mesh(self, mesh_id: int) -> bool:
        request = VCCSim_pb2.MeshID(id=mesh_id)
        response = self.mesh_service.RemoveGlobalMesh(request)
        return response.status

    # Point Cloud Service Methods
    def send_point_cloud_with_color(self, points: List[Tuple[float, float, float]], colors: List[int]) -> bool:
        if len(points) != len(colors):
            raise ValueError("Number of points must match number of colors")

        colored_points = []
        for (x, y, z), color in zip(points, colors):
            point = self._create_vec3(x, y, z)
            colored_points.append(VCCSim_pb2.ColoredPoint(point=point, color=color))

        request = VCCSim_pb2.ColoredPointCloud(data=colored_points)
        response = self.point_cloud_service.SendPointCloudWithColor(request)
        return response.status

    # Safe Check Service Methods
    def check_safety_pawn(self, robot_name: str) -> bool:
        request = self._create_robot_name(robot_name)
        response = self.safe_check_service.CheckSafetyPawn(request)
        return response.status

    def check_safety_position(self, x: float, y: float, z: float) -> bool:
        request = self._create_vec3(x, y, z)
        response = self.safe_check_service.CheckSafetyPosition(request)
        return response.status

    def check_safety_drone_path(self, name: str, poses: List[Tuple[float, float, float, float, float, float]]) -> bool:
        path = [self._create_pose(*pose) for pose in poses]
        request = VCCSim_pb2.DronePath(name=name, path=path)
        response = self.safe_check_service.CheckSafetyDronePath(request)
        return response.status


class RGBImageUtils:
    """Utility class for handling RGB-like image data from VCCSim."""

    @staticmethod
    def process_rgb_image_data(image_data: VCCSim_pb2.RGBData) -> np.ndarray:
        width = image_data.width
        height = image_data.height
        raw = np.frombuffer(image_data.data, dtype=np.uint8)

        expected_size = width * height * 4
        if raw.size != expected_size:
            raise ValueError(f"Unexpected RGB buffer size: {raw.size} vs expected {expected_size}")

        bgra = raw.reshape((height, width, 4))
        return bgra[..., :3][..., ::-1].copy()

    @staticmethod
    def save_rgb_image(image_data: VCCSim_pb2.RGBData, output_path: str) -> bool:
        try:
            rgb_array = RGBImageUtils.process_rgb_image_data(image_data)
            image = Image.fromarray(rgb_array)
            image.save(output_path)
            return True
        except Exception as exc:
            print(f"Error saving image: {exc}")
            return False

    @staticmethod
    def convert_to_cv2_format(image_data: VCCSim_pb2.RGBData) -> np.ndarray:
        rgb_array = RGBImageUtils.process_rgb_image_data(image_data)
        return rgb_array[..., ::-1].copy()

    @staticmethod
    def save_depth_image(depth_data: VCCSim_pb2.DepthData, output_path: str) -> bool:
        try:
            width, height = depth_data.width, depth_data.height
            buffer = np.array(depth_data.data, dtype=np.float32)
            expected_size = width * height
            if buffer.size != expected_size:
                raise ValueError(f"Unexpected depth buffer size: {buffer.size} vs expected {expected_size}")

            image = buffer.reshape((height, width))
            finite_mask = np.isfinite(image)
            if finite_mask.any():
                valid = image[finite_mask]
                min_val = valid.min()
                max_val = valid.max()
                if max_val > min_val:
                    scaled = (image - min_val) / (max_val - min_val)
                else:
                    scaled = np.zeros_like(image, dtype=np.float32)
            else:
                scaled = np.zeros_like(image, dtype=np.float32)

            image_8bit = (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(image_8bit, mode="L").save(output_path)
            return True
        except Exception as exc:
            print(f"Error saving depth image: {exc}")
            return False

    @staticmethod
    def save_segmentation_image(seg_data: VCCSim_pb2.RGBData, output_path: str) -> bool:
        try:
            rgb_array = RGBImageUtils.process_rgb_image_data(seg_data)
            image = Image.fromarray(rgb_array)
            image.save(output_path)
            return True
        except Exception as exc:
            print(f"Error saving segmentation image: {exc}")
            return False

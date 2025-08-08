import grpc
from . import VCCSim_pb2
from . import VCCSim_pb2_grpc
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import io

class VCCSimClient:
    """Client for interacting with VCCSim services."""
    
    def __init__(self, host: str = "localhost", port: int = 50996, max_message_length: int = 20 * 1024 * 1024):
        """Initialize the VCCSim client.
        
        Args:
            host: Server hostname
            port: Server port number
            max_message_length: Maximum message length in bytes (default: 20MB)
        """
        # Configure channel options to handle larger messages
        options = [
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
        
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)
        
        # Initialize service stubs
        self.record_service = VCCSim_pb2_grpc.RecordingServiceStub(self.channel)
        self.lidar_service = VCCSim_pb2_grpc.LidarServiceStub(self.channel)
        self.depth_camera_service = VCCSim_pb2_grpc.DepthCameraServiceStub(self.channel)
        self.rgb_camera_service = VCCSim_pb2_grpc.RGBCameraServiceStub(self.channel)
        self.drone_service = VCCSim_pb2_grpc.DroneServiceStub(self.channel)
        self.car_service = VCCSim_pb2_grpc.CarServiceStub(self.channel)
        self.flash_service = VCCSim_pb2_grpc.FlashServiceStub(self.channel)
        self.mesh_service = VCCSim_pb2_grpc.MeshServiceStub(self.channel)
        self.point_cloud_service = VCCSim_pb2_grpc.PointCloudServiceStub(self.channel)

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()

    # Helper methods
    def _create_robot_name(self, name: str) -> VCCSim_pb2.RobotName:
        """Create a RobotName message."""
        return VCCSim_pb2.RobotName(name=name)

    def _create_pose(self, x: float, y: float, z: float, 
                    roll: float, pitch: float, yaw: float) -> VCCSim_pb2.Pose:
        """Create a Pose message."""
        return VCCSim_pb2.Pose(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw
        )

    def _create_pose_only_yaw(self, x: float, y: float, z: float, 
                             yaw: float) -> VCCSim_pb2.PoseOnlyYaw:
        """Create a PoseOnlyYaw message."""
        return VCCSim_pb2.PoseOnlyYaw(x=x, y=y, z=z, yaw=yaw)
    
    # Record Service Methods
    def toggle_recording(self) -> bool:
        response = self.record_service.Recording(VCCSim_pb2.EmptyRequest())
        return response.status

    # LiDAR Service Methods
    def get_lidar_data(self, robot_name: str) -> List[Tuple[float, float, float]]:
        """Get LiDAR data for a robot.
        
        Returns:
            List of (x, y, z) tuples representing LiDAR points
        """
        request = self._create_robot_name(robot_name)
        response = self.lidar_service.GetLiDARData(request)
        return [(point.x, point.y, point.z) for point in response.data]

    def get_lidar_odom(self, robot_name: str) -> Tuple[VCCSim_pb2.Pose, VCCSim_pb2.twist]:
        """Get LiDAR odometry for a robot."""
        request = self._create_robot_name(robot_name)
        response = self.lidar_service.GetLiDAROdom(request)
        return response.pose, response.twist

    def get_lidar_data_and_odom(self, robot_name: str) -> Tuple[List[Tuple[float, float, float]], VCCSim_pb2.Odometry]:
        """Get both LiDAR data and odometry for a robot."""
        request = self._create_robot_name(robot_name)
        response = self.lidar_service.GetLiDARDataAndOdom(request)
        points = [(point.x, point.y, point.z) for point in response.data.data]
        return points, response.odom

    # Depth Camera Service Methods
    def get_depth_camera_point_data(self, robot_name: str, index: int) -> List[Tuple[float, float, float]]:
        """Get depth camera point data for a robot."""
        request = VCCSim_pb2.IndexedCamera(
            robot_name=self._create_robot_name(robot_name), 
            index=index
            )
        response = self.depth_camera_service.GetDepthIndexedCameraPointData(request)
        return [(point.x, point.y, point.z) for point in response.data]

    def get_depth_camera_image_size(self, robot_name: str, index: int) -> Tuple[int, int]:
        """Get depth camera image size (width, height) for a robot."""
        request = VCCSim_pb2.IndexedCamera(
            robot_name=self._create_robot_name(robot_name), 
            index=index
            )
        response = self.depth_camera_service.GetDepthIndexedCameraImageSize(request)
        return response.width, response.height

    def get_depth_camera_image_data(self, robot_name: str, index: int) -> List[float]:
        """Get depth camera image data for a robot."""
        request = VCCSim_pb2.IndexedCamera(
            robot_name = self._create_robot_name(robot_name), 
            index=index
            )
        response = self.depth_camera_service.GetDepthIndexedCameraImageData(request)
        return list(response.data)

    def get_depth_camera_odom(self, robot_name: str) -> VCCSim_pb2.Odometry:
        """Get depth camera odometry for a robot."""
        request = self._create_robot_name(robot_name)
        return self.depth_camera_service.GetDepthCameraOdom(request)

    # RGB Camera Service Methods
    def get_rgb_camera_odom(self, robot_name: str) -> VCCSim_pb2.Odometry:
        """Get RGB camera odometry for a robot."""
        request = self._create_robot_name(robot_name)
        return self.rgb_camera_service.GetRGBCameraOdom(request)
    
    def get_rgb_indexed_camera_image_size(self, robot_name: str, index: int) -> Tuple[int, int]:
        """Get RGB camera image size for a specific camera index.
        
        Args:
            robot_name: The name of the robot
            index: Camera index
        
        Returns:
            Tuple of width and height
        """
        request = VCCSim_pb2.IndexedCamera(
            robot_name=self._create_robot_name(robot_name),
            index=index
        )
        response = self.rgb_camera_service.GetRGBIndexedCameraImageSize(request)
        return response.width, response.height

    def get_rgb_indexed_camera_image_data(self, robot_name: str, index: int, 
                                         format: VCCSim_pb2.Format = VCCSim_pb2.Format.PNG) -> VCCSim_pb2.RGBCameraImageData:
        """Get RGB camera image data for a specific camera index.
        
        Args:
            robot_name: The name of the robot
            index: Camera index
            format: Image format (PNG, JPEG, or RAW). Default is PNG.
        
        Returns:
            RGBCameraImageData object with width, height, data, format and optional metadata
        """
        request = VCCSim_pb2.IndexedCamera(
            robot_name=self._create_robot_name(robot_name),
            index=index,
            format=format
        )
        return self.rgb_camera_service.GetRGBIndexedCameraImageData(request)
    
    def save_rgb_image(self, image_data: VCCSim_pb2.RGBCameraImageData, 
                      output_path: str) -> bool:
        """Save RGB camera image data to a file.
        
        Args:
            image_data: RGBCameraImageData object 
            output_path: File path to save the image
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            # Open file in binary write mode
            with open(output_path, 'wb') as f:
                # Write raw data directly to file
                f.write(image_data.data)
                
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    # Drone Service Methods
    def get_drone_pose(self, robot_name: str) -> VCCSim_pb2.Pose:
        """Get drone pose."""
        request = self._create_robot_name(robot_name)
        return self.drone_service.GetDronePose(request)

    def send_drone_pose(self, name: str, x: float, y: float, z: float,
                       roll: float, pitch: float, yaw: float) -> bool:
        """Send drone pose."""
        pose = self._create_pose(x, y, z, roll, pitch, yaw)
        request = VCCSim_pb2.DronePose(name=name, pose=pose)
        response = self.drone_service.SendDronePose(request)
        return response.status

    def send_drone_path(self, name: str, poses: List[Tuple[float, float, float, float, float, float]]) -> bool:
        """Send drone path as a list of poses."""
        path_poses = [self._create_pose(*pose) for pose in poses]
        request = VCCSim_pb2.DronePath(name=name, path=path_poses)
        response = self.drone_service.SendDronePath(request)
        return response.status

    # Car Service Methods
    def get_car_odom(self, robot_name: str) -> VCCSim_pb2.Odometry:
        """Get car odometry."""
        request = self._create_robot_name(robot_name)
        return self.car_service.GetCarOdom(request)

    def send_car_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> bool:
        """Send car pose."""
        pose = self._create_pose_only_yaw(x, y, z, yaw)
        request = VCCSim_pb2.CarPose(name=name, pose=pose)
        response = self.car_service.SendCarPose(request)
        return response.status

    def send_car_path(self, name: str, poses: List[Tuple[float, float, float, float]]) -> bool:
        """Send car path as a list of poses (x, y, z, yaw)."""
        path_poses = [self._create_pose_only_yaw(*pose) for pose in poses]
        request = VCCSim_pb2.CarPath(name=name, path=path_poses)
        response = self.car_service.SendCarPath(request)
        return response.status
        
    # Flash Service Methods
    def get_flash_pose(self, robot_name: str) -> VCCSim_pb2.Pose:
        """Get flash pose."""
        request = self._create_robot_name(robot_name)
        return self.flash_service.GetFlashPose(request)
        
    def send_flash_pose(self, name: str, x: float, y: float, z: float,
                       roll: float, pitch: float, yaw: float) -> bool:
        """Send flash pose."""
        pose = self._create_pose(x, y, z, roll, pitch, yaw)
        request = VCCSim_pb2.FlashPose(name=name, pose=pose)
        response = self.flash_service.SendFlashPose(request)
        return response.status
        
    def send_flash_path(self, name: str, poses: List[Tuple[float, float, float, float, float, float]]) -> bool:
        """Send flash path as a list of poses."""
        path_poses = [self._create_pose(*pose) for pose in poses]
        request = VCCSim_pb2.FlashPath(name=name, path=path_poses)
        response = self.flash_service.SendFlashPath(request)
        return response.status
        
    def check_flash_ready(self, robot_name: str) -> bool:
        """Check if flash is ready."""
        request = self._create_robot_name(robot_name)
        response = self.flash_service.CheckFlashReady(request)
        return response.status
        
    def move_to_next(self, robot_name: str) -> bool:
        """Move flash to next position."""
        request = self._create_robot_name(robot_name)
        response = self.flash_service.MoveToNext(request)
        return response.status

    # Mesh Service Methods
    def send_mesh(self, data: bytes, format: int, version: int, simplified: bool,
                 transform_pose: Tuple[float, float, float, float, float, float]) -> bool:
        """Send mesh data."""
        transform = self._create_pose(*transform_pose)
        request = VCCSim_pb2.MeshData(
            data=data,
            format=format,
            version=version,
            simplified=simplified,
            transform=transform
        )
        response = self.mesh_service.SendMesh(request)
        return response.status
        
    def send_global_mesh(self, data: bytes, format: int, version: int, simplified: bool,
                        transform_pose: Tuple[float, float, float, float, float, float]) -> int:
        """Send global mesh data.
        
        Returns:
            The id of the created mesh
        """
        transform = self._create_pose(*transform_pose)
        request = VCCSim_pb2.MeshData(
            data=data,
            format=format,
            version=version,
            simplified=simplified,
            transform=transform
        )
        response = self.mesh_service.SendGlobalMesh(request)
        return response.id
        
    def remove_global_mesh(self, mesh_id: int) -> bool:
        """Remove a global mesh by its ID.
        
        Args:
            mesh_id: The ID of the mesh to remove
            
        Returns:
            True if removal was successful, False otherwise
        """
        request = VCCSim_pb2.MeshID(id=mesh_id)
        response = self.mesh_service.RemoveGlobalMesh(request)
        return response.status

    # Point Cloud Service Methods
    def send_point_cloud_with_color(self, points: List[Tuple[float, float, float]], 
                                  colors: List[int]) -> bool:
        """Send colored point cloud data."""
        if len(points) != len(colors):
            raise ValueError("Number of points must match number of colors")

        point_data = []
        for (x, y, z), color in zip(points, colors):
            point = VCCSim_pb2.Point(x=x, y=y, z=z)
            point_with_color = VCCSim_pb2.PointWithColor(point=point, color=color)
            point_data.append(point_with_color)

        request = VCCSim_pb2.PointCloudWithColor(data=point_data)
        response = self.point_cloud_service.SendPointCloudWithColor(request)
        return response.status
    
class RGBImageUtils:
    """Utility class for handling RGB image data from VCCSim."""
    
    @staticmethod
    def process_rgb_image_data(image_data: VCCSim_pb2.RGBCameraImageData) -> np.ndarray:
        """Process RGB camera image data into a NumPy array.
        
        Args:
            image_data: RGBCameraImageData object from the gRPC call
            
        Returns:
            NumPy array of the image in RGB format (height, width, 3)
        """
        # Handle different image formats
        if image_data.format == VCCSim_pb2.Format.PNG or image_data.format == VCCSim_pb2.Format.JPEG:
            # Handle compressed formats using PIL
            img = Image.open(io.BytesIO(image_data.data))
            return np.array(img)
        
        elif image_data.format == VCCSim_pb2.Format.RAW:
            # Handle raw format (RGB byte array)
            width = image_data.width
            height = image_data.height
            bytes_per_pixel = image_data.bytes_per_pixel if image_data.bytes_per_pixel else 3
            
            # Reshape the flat byte array into a 3D array (height, width, channels)
            img_array = np.frombuffer(image_data.data, dtype=np.uint8)
            return img_array.reshape((height, width, bytes_per_pixel))
        
        else:
            raise ValueError(f"Unsupported image format: {image_data.format}")
    
    @staticmethod
    def save_rgb_image(image_data: VCCSim_pb2.RGBCameraImageData, output_path: str) -> bool:
        """Save RGB camera image data to a file.
        
        Args:
            image_data: RGBCameraImageData object 
            output_path: File path to save the image
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            # For PNG and JPEG, we can write directly
            if image_data.format in [VCCSim_pb2.Format.PNG, VCCSim_pb2.Format.JPEG]:
                with open(output_path, 'wb') as f:
                    f.write(image_data.data)
                return True
            
            # For RAW format, convert to a standard image format
            elif image_data.format == VCCSim_pb2.Format.RAW:
                img_array = RGBImageUtils.process_rgb_image_data(image_data)
                img = Image.fromarray(img_array)
                img.save(output_path)
                return True
                
            else:
                raise ValueError(f"Unsupported image format: {image_data.format}")
                
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def convert_to_cv2_format(image_data: VCCSim_pb2.RGBCameraImageData) -> np.ndarray:
        """Convert RGB camera image data to OpenCV compatible format (BGR).
        
        Args:
            image_data: RGBCameraImageData object
            
        Returns:
            NumPy array in BGR format suitable for OpenCV functions
        """
        # First get the RGB numpy array
        rgb_array = RGBImageUtils.process_rgb_image_data(image_data)
        
        # Convert RGB to BGR (OpenCV uses BGR)
        return rgb_array[..., ::-1].copy()  # Reverse the color channel order
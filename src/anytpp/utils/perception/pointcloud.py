import numpy as np
from numpy import typing as npt
import open3d as o3d


def depth_image_to_pointcloud(
    depth_image: npt.NDArray[np.float32], camera_intrinsics: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Converts a depth image to a pointcloud using the camera intrinsics.

    Args:
        depth_image (npt.NDArray[np.float32]): The depth image.
        camera_intrinsics (npt.NDArray[np.float32]): The camera intrinsic matrix of the RGBD camera.

    Returns:
        npt.NDArray[np.float32]: The pointcloud in the camera coordinates.
    """
    # Get the height and width of the depth image
    height, width = depth_image.shape

    # Extract focal lengths (fx, fy) and principal point (cx, cy) from the intrinsics matrix
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Create an Open3D PinholeCameraIntrinsic object
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Convert the NumPy depth image to an Open3D Image object
    o3d_depth_image = o3d.geometry.Image(depth_image)

    # Create a point cloud from the depth image and camera intrinsics
    # This is a highly optimized Open3D function.
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth_image, o3d_intrinsics
    )

    return np.asarray(pcd.points, dtype=np.float32)


def to_o3d_pointcloud_gpu(
    pointcloud_np: npt.NDArray[np.float32],
) -> o3d.t.geometry.PointCloud:
    """
    Converts a NumPy pointcloud to a GPU-based Open3D tensor pointcloud.

    Args:
        pointcloud_np (npt.NDArray[np.float32]): The numpy pointcloud
            to convert. Shape: (N, 3)

    Returns:
        o3d.t.geometry.PointCloud: The Open3D pointcloud object residing on the GPU.
    """
    # Define the GPU device
    device = o3d.core.Device("CUDA:0")

    # Create a tensor-based pointcloud on the specified device
    pointcloud_o3d = o3d.t.geometry.PointCloud(device)

    # Load the numpy pointcloud onto the GPU
    pointcloud_o3d.point.positions = o3d.core.Tensor(pointcloud_np, device=device)

    return pointcloud_o3d


def get_valid_pointcloud(
    pointcloud: o3d.t.geometry.PointCloud,
) -> o3d.t.geometry.PointCloud:
    return (
        to_o3d_pointcloud_gpu(pointcloud)
        .remove_non_finite_points(remove_nan=True, remove_infinite=True)
        .cpu()
        .numpy()
    )


def calculate_surface_normals(
    pointcloud: npt.NDArray[np.float32],
    radius: float = 0.1,
    max_nn: int = 30,
) -> npt.NDArray[np.float32]:
    """
    Calculates surface normals on the GPU using a CUDA-enabled Open3D.

    Args:
        pointcloud (npt.NDArray[np.float32]): The pointcloud to calculate
            surface normals for. Shape: (N, 3)
        radius (float): The radius used to find neighboring points.
        max_nn (int): The maximum number of neighbors to search for.

    Returns:
        npt.NDArray[np.float32]: The calculated surface normal vectors,
            with a shape of (N, 3).
    """
    # Convert numpy pointcloud to open3d pointcloud on GPU
    pcd_gpu = to_o3d_pointcloud_gpu(pointcloud_np=pointcloud)

    # Estimate normal vectors
    pcd_gpu.estimate_normals(max_nn=max_nn, radius=radius)

    # Normalize all normals to have length = 1
    pcd_gpu.normalize_normals()

    # Orient normals consistently (since there is no "in" or "out")
    camera_location = o3d.core.Tensor(
        [0, 0, 0], device=pcd_gpu.device, dtype=o3d.core.Dtype.Float32
    )
    pcd_gpu.orient_normals_towards_camera_location(camera_location)

    # Offload to CPU and convert to numpy array
    return pcd_gpu.point.normals.cpu().numpy()


def get_pixel_coordinates_from_pointcloud(
    pointcloud: npt.NDArray[np.float32], camera_intrinsics: npt.NDArray[np.float32]
) -> npt.NDArray[np.int_]:
    """
    Calculates the pixel coordinates of the pointcloud points in the image,
    using the camera intrinsic matrix.

    Args:
        pointcloud (npt.NDArray[np.float32]): The pointcloud. Shape: `(N, 3)`
        camera_intrinsics (npt.NDArray[np.float32]): The camera intrinsic matrix. Shape: `(3, 3)`

    Returns:
        npt.NDArray[np.int]: The pixel coordinates of every point in the pointcloud.
    """
    fx, fy, cx, cy = camera_intrinsics[[0, 0], [1, 1], [0, 2], [1, 2]]
    x, y, z = pointcloud.T
    pixel_coordinates: npt.NDArray = (
        np.array([[fx, fy]]) * np.vstack([x, y]) / z
    ) + np.array([[cx, cy]]).T
    return pixel_coordinates.astype(dtype=np.int_)

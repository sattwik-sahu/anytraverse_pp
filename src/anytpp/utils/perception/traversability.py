from dataclasses import asdict, dataclass

import numpy as np
from anytraverse import AnyTraverse
from anytraverse.utils.pipelines.base import AnyTraverseState
from numpy import typing as npt

from anytpp.utils.perception.pointcloud import (
    calculate_surface_normals,
    get_pixel_coordinates_from_pointcloud,
    get_valid_pointcloud,
)


@dataclass
class AnyTraversePlusPlusState(AnyTraverseState):
    surface_normals: npt.NDArray[np.float32]
    slope_scores: npt.NDArray[np.float32]
    valid_pointcloud: npt.NDArray[np.float32]
    traversability_scores: npt.NDArray[np.float32]


class AnyTraversePlusPlusTraversabilitySegmentation:
    def __init__(
        self, anytraverse: AnyTraverse, normal_radius: float, normal_max_nn: int
    ) -> None:
        self._anytraverse = anytraverse
        self._normal_radius = normal_radius
        self._normal_max_nn = normal_max_nn

    def step(
        self,
        image: npt.NDArray[np.uint8],
        pointcloud: npt.NDArray[np.float32],
        camera_extrinsics: npt.NDArray[np.float32],
        camera_intrinsics: npt.NDArray[np.float32],
    ) -> AnyTraversePlusPlusState:
        anytraverse_state = self._anytraverse.step(image=image)
        traversability_map = anytraverse_state.traversability_map

        valid_pointcloud = get_valid_pointcloud(pointcloud=pointcloud)
        pointcloud_pixel_indexes = get_pixel_coordinates_from_pointcloud(
            pointcloud=pointcloud, camera_intrinsics=camera_intrinsics
        )
        surface_normals = (
            calculate_surface_normals(
                pointcloud=valid_pointcloud,
                radius=self._normal_radius,
                max_nn=self._normal_max_nn,
            )
            @ camera_extrinsics[:3, :3]
        )
        pointcloud_slope_scores = np.abs(surface_normals[:, 2])
        pointcloud_semantic_traversability_scores = traversability_map[
            pointcloud_pixel_indexes
        ]
        pointcloud_traversability_scores = (
            pointcloud_slope_scores * pointcloud_semantic_traversability_scores.numpy()
        )

        return AnyTraversePlusPlusState(
            **asdict(anytraverse_state),
            surface_normals=surface_normals,
            slope_scores=pointcloud_slope_scores,
            valid_pointcloud=valid_pointcloud,
            traversability_scores=pointcloud_traversability_scores,
        )

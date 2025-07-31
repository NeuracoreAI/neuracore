from neuracore.core.nc_types import PointCloudData


def test_point_cloud_encoding():
    point_cloud = PointCloudData(points=[[0, 0, 0], [1, 1, 1]])
    assert isinstance(point_cloud.model_dump()["points"], str)

    round_trip = PointCloudData.model_validate_json(point_cloud.model_dump_json())
    assert round_trip.points[0][0] == point_cloud.points[0][0]

    round_trip = PointCloudData.model_validate(point_cloud.model_dump())

    assert round_trip.points[0][0] == point_cloud.points[0][0]

    assert (
        PointCloudData.model_validate(
            {"points": [[0, 0, 0], [1, 1, 1]], "timestamp": 100}
        ).points[0][0]
        == 0
    )

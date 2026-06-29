from perception_service.object_parser import parse_grounded_objects


def test_parse_grounded_objects_clamps_bbox_and_filters_confidence():
    objects = parse_grounded_objects(
        {
            "objects": [
                {"label": "banana", "bbox_2d": [-10, 20, 700, 500], "confidence": 0.9},
                {"label": "cup", "bbox_2d": [10, 10, 20, 20], "confidence": 0.1},
                {"label": "bad", "bbox_2d": [5, 5, 5, 10], "confidence": 1.0},
            ]
        },
        image_width=640,
        image_height=480,
        min_confidence=0.5,
    )

    assert len(objects) == 1
    assert objects[0].label == "banana"
    assert objects[0].bbox_xyxy == (0, 20, 639, 479)


def test_parse_grounded_objects_accepts_xywh_dict_bbox():
    objects = parse_grounded_objects(
        {"objects": [{"name": "block", "bbox": {"x": 10, "y": 20, "width": 30, "height": 40}}]},
        image_width=100,
        image_height=100,
    )

    assert len(objects) == 1
    assert objects[0].bbox_xyxy == (10, 20, 40, 60)


def test_parse_grounded_objects_scales_1000_grid_bbox_to_image_pixels():
    objects = parse_grounded_objects(
        {"objects": [{"label": "strawberry", "bbox_2d": [554, 31, 856, 529], "confidence": 0.9}]},
        image_width=640,
        image_height=480,
    )

    assert len(objects) == 1
    assert objects[0].bbox_xyxy == (355, 15, 548, 254)

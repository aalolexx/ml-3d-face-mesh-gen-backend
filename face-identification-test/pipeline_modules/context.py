from dataclasses import dataclass
from typing import Dict, List
from pipeline_util.enums import ComparisonMethod

@dataclass
class TestingResultEntry:
    open_testing_entry_id: int
    method: str  # Pandas and pkl cant seem to not be able to serialize an enum. so save key here
    prediction: float


@dataclass
class OpenTestingEntry:
    # id --> dict key
    gallery_image_file_name: str
    input_image_file_name: str
    is_actual_match: int
    # Additional Infos:
    rotation_angle: int
    scenario: str


@dataclass
class FailedTestingEntry:
    failed_method: str
    open_testing_entry_id: int
    fail_reason: str


@dataclass
class Context:
    input_dir_path: str
    output_dir_path: str
    working_dir_path: str
    misc_dir_path: str
    deep_3d_coeffs: dict
    face_recognition_2d_encodings: dict
    open_testing_entry: Dict[int, OpenTestingEntry]
    testing_result_entries: List[TestingResultEntry]
    failed_testing_entries: List[FailedTestingEntry]
    panda_testing_entries: any
    panda_failed_entries: any

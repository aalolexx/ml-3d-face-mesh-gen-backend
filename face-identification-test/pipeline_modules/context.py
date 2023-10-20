from dataclasses import dataclass
from typing import Dict, List

@dataclass
class TestingResultEntry:
    open_testing_entry_id: int
    method: str
    prediction: float


@dataclass
class OpenTestingEntry:
    # id --> dict key
    gallery_image_file_name: str
    input_image_file_name: str
    is_actual_match: bool


@dataclass
class Context:
    input_dir_path: str
    output_dir_path: str
    working_dir_path: str
    misc_dir_path: str
    deep_3d_coeffs: dict
    face_recon_2d_encodings: dict
    open_testing_entry: Dict[int, OpenTestingEntry]
    testing_result_entries: List[TestingResultEntry]

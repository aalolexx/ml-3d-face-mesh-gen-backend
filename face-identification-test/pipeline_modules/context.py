from dataclasses import dataclass
from typing import List

@dataclass
class TestingEntry:
    id: int
    method: str
    gallery_image_file_name: str
    input_image_file_name: str
    is_actual_match: bool
    prediction: float


@dataclass
class Context:
    input_dir_path: str
    output_dir_path: str
    working_dir_path: str
    misc_dir_path: str
    deep_3d_coeffs: dict
    testing_entries: List[TestingEntry]

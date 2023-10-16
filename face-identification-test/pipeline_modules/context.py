from dataclasses import dataclass

@dataclass
class Context:
    input_dir_path: str
    output_dir_path: str
    working_dir_path: str
    misc_dir_path: str
    deep_3d_coeffs: dict
from enum import Enum


class ComparisonMethod:
    def __init__(self, name, title, color):
        self.name = name
        self.title = title
        self.color = color

    def __str__(self):
        return self.name


class ComparisonMethods(Enum):
    @property
    def name(self):
        return self.value.name

    @property
    def title(self):
        return self.value.title

    @property
    def color(self):
        return self.value.color

    FACE_RECOGNITION_DISTANCE_2D = ComparisonMethod(
        'face_recognition_distance_2d',
        'face_recognition 2D',
        'indigo'
    )
    DEEPFACE_DISTANCE_2D = ComparisonMethod(
        'deepface_distance_2d',
        'DeepFace 2D',
        'slateblue'
    )
    COEFFICIENT_BASED_3D = ComparisonMethod(
        'coefficient_based_3d',
        'coeff distance 3DMM',
        'seagreen'
    )
    BIDIRECTIONAL_VPN_COMPARE = ComparisonMethod(
        'bidirectional_vpn_compare',
        'VPN Image Compare (bi)',
        'mediumaquamarine'
    )

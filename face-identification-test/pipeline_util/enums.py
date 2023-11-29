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
        return self.value.color  # used for plotting

    FACE_RECOGNITION_DISTANCE_2D = ComparisonMethod(
        'face_recognition_distance_2d',
        'face_recognition 2D',
        '#e261c7'
    )
    DEEPFACE_DISTANCE_2D_VGG = ComparisonMethod(
        'deepface_distance_2d_vgg',
        'DeepFace 2D w VGG-Face',
        '#b2298b'
    )
    DEEPFACE_DISTANCE_2D_FACENET = ComparisonMethod(
        'deepface_distance_2d_facenet',
        'DeepFace 2D w Facenet',
        '#730053'
    )
    COEFFICIENT_BASED_3D = ComparisonMethod(
        'coefficient_based_3d',
        'coeff distance 3DMM',
        '#88c7fe'
    )
    BIDIRECTIONAL_VPN_COMPARE = ComparisonMethod(
        'bidirectional_vpn_compare',
        'VPN Image Compare (bi) w face_recognition',
        '#4094e1'
    )
    UNIDIRECTIONAL_VPN_COMPARE = ComparisonMethod(
        'unidirectional_vpn_compare',
        'VPN Image Compare (uni) w face_recognition',
        '#1161b5'
    )


class ComparisonFramework(Enum):
    FACE_RECOGNITION = 'face_recognition'
    DEEPFACE = 'deepface'


# DATASET and max entries
class Datasets(Enum):
    YALE = 15
    LFW = 500
    MULTIPIE = 250

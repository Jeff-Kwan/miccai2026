from .depth_attenuation import DepthAttenuationVideo
from .haze_artifact import VideoHazeArtifact
from .gaussian_shadow import GaussianShadowVideo
from .speckle_reduction import VideoSpeckleReduction

from torchvision.transforms import v2

def get_pretrain_augmentations():
    return v2.Compose([
        DepthAttenuationVideo(attenuation_rate=(0.5, 2.0), max_attenuation=0.1, p=0.4),
        GaussianShadowVideo(strength=(0.3, 0.7), sigma_x=(0.01, 0.2), sigma_y=(0.01, 0.2), p=0.3),
        VideoHazeArtifact(radius=(0.05, 0.5), sigma=(0.01, 0.05), p=0.2),
        VideoSpeckleReduction(sigma_spatial=(0.05, 0.5), sigma_color=(0.05, 0.5), p=0.1),
        ])
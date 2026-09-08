from .redcnn import REDCNN
from .wgan_vgg import UNetGen, PatchCritic, VGGPerceptual, gradient_penalty
from .dugan import DUGAN
from .ctformer import CTformer


def build_baseline(name: str, **kw):
    name = name.lower()
    if name == "redcnn":
        return REDCNN()
    if name == "wgan_vgg":
        return UNetGen()
    if name == "dugan":
        return DUGAN()
    if name == "ctformer":
        return CTformer(**kw)
    raise KeyError(name)

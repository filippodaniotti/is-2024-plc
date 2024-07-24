from .tilt_loss import TiltLoss, TiltLossBuilder
from . import ObjectFactory

__all__ = [
    "TiltLoss",
    "loss_factory",
]

loss_factory = ObjectFactory()
loss_factory.register_builder("tilt_loss", TiltLossBuilder())

    
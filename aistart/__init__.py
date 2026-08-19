"""AI Start Expert — generative starting-point lens design for CODE V.

Generates a viable first-pass lens design (as a CODE V .seq file) from a small
set of specifications (focal length, field of view, aperture), entirely
on-premise.
"""
from .specs import Spec
from .generator import generate, Result
from .catalog import CATALOG, get as get_prototype
from . import codevexport

__version__ = "1.0.0"

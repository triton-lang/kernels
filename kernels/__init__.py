# from .conv import _conv, conv
from . import blocksparse
from .cross_entropy import _cross_entropy, cross_entropy
from .flash_attention import attention
from .matmul import _matmul, get_higher_dtype, matmul

# SM120 Gluon flash attention requires triton.experimental.gluon (SM12x only)
try:
    from .flash_attention_sm120 import attention_forward_sm120
except ImportError:
    attention_forward_sm120 = None

__all__ = [
    "blocksparse",
    "_cross_entropy",
    "cross_entropy",
    "_matmul",
    "matmul",
    "attention",
    "attention_forward_sm120",
    "get_higher_dtype",
]

from typing import Literal, Tuple, get_args

NoiseLevel = Literal[20, 50, 75]
NoiseType = Literal['permute', 'open', 'mix']

NOISE_LEVELS: Tuple[NoiseLevel] = get_args(NoiseLevel)
NOISE_TYPES: Tuple[NoiseType] = get_args(NoiseType)

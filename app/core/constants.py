from typing import Final

N_FRAMES: Final[int] = 16

IMG_SIZE: Final[int] = 224

MEAN: Final[tuple[float, float, float]] = (
    0.485,
    0.456,
    0.406,
)

STD: Final[tuple[float, float, float]] = (
    0.229,
    0.224,
    0.225,
)

CRASH_THRESHOLD: Final[float] = 0.50
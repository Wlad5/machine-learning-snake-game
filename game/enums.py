from enum import Enum
class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GameStatus(Enum):
    RUNNING = 0
    PAUSED = 1
    GAME_OVER = 2
    WIN = 3

class CellType(Enum):
    EMPTY = 0
    SNAKE = 1
    FOOD = 2
    WALL = 3

class EventType(Enum):
    ATE_FOOD = 0
    HIT_WALL = 1
    HIT_SELF = 2
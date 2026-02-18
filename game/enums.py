from enum import Enum
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Action(Enum):
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STRAIGHT = 3
    MOVE_UP = 4
    MOVE_DOWN = 5

class GameStatus(Enum):
    RUNNING = 1
    PAUSED = 2
    GAME_OVER = 3
    WIN = 4

class CellType(Enum):
    EMPTY = 0
    SNAKE = 1
    FOOD = 3
    WALL = 4

class EventType(Enum):
    ATE_FOOD = 1
    HIT_WALL = 2
    HIT_SELF = 3
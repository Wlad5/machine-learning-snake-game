import pygame as pg
from enums import Direction, Action, GameStatus, CellType, EventType

class InputHandler:
    # Maps keys (or later agent actions) to movement commands.
    def handle_event(self, event):
        if event.type == pg.KEYDOWN:
            match event.key:
                case pg.K_UP:
                    return Direction.UP
                case pg.K_DOWN:
                    return Direction.DOWN
                case pg.K_LEFT:
                    return Direction.LEFT
                case pg.K_RIGHT:
                    return Direction.RIGHT
        return None
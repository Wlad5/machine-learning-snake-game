import pygame as pg
from constants import FOOD_COLOR, CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT
from enums import Direction, Action, GameStatus, CellType, EventType

class Snake:
    # Manages body segments, movement, growth, and self‑collision checks.
    def __init__(self):
        self.snake = [((SCREEN_WIDTH // CELL_SIZE // 2) * CELL_SIZE, (SCREEN_HEIGHT // CELL_SIZE // 2) * CELL_SIZE)]
        self.snake_head = self.snake[0]
        self.skin = pg.Surface((CELL_SIZE, CELL_SIZE))
        self.skin.fill((255, 255, 255))
    
    def draw_snake(self, screen):
        for segment in self.snake:
            screen.blit(self.skin, segment)

    def move(self, direction):
        match direction:
            case Direction.UP:
                new_head = (self.snake_head[0], self.snake_head[1] - CELL_SIZE)

            case Direction.DOWN:
                new_head = (self.snake_head[0], self.snake_head[1] + CELL_SIZE)

            case Direction.LEFT:
                new_head = (self.snake_head[0] - CELL_SIZE, self.snake_head[1])

            case Direction.RIGHT:
                new_head = (self.snake_head[0] + CELL_SIZE, self.snake_head[1])

            case _:
                return
        self.snake.insert(0, new_head)
        self.snake.pop()
        self.snake_head = new_head
    
    def grow(self):
        tail = self.snake[-1]
        self.snake.append(tail)
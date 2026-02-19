import pygame as pg
from constants import CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, SNAKE_HEAD_COLOR
from enums import Direction

class Snake:
    # Manages body segments, movement, growth, and self‑collision checks.
    def __init__(self):
        self.snake = [((SCREEN_WIDTH // CELL_SIZE // 2) * CELL_SIZE, (SCREEN_HEIGHT // CELL_SIZE // 2) * CELL_SIZE)]
        self.snake_head = self.snake[0]
        self.skin = pg.Surface((CELL_SIZE, CELL_SIZE))
        self.skin.fill((255, 255, 255))
        self.head_skin = pg.Surface((CELL_SIZE, CELL_SIZE))
        self.head_skin.fill((SNAKE_HEAD_COLOR))
    
    def draw_snake(self, screen):
        for segment in self.snake:
            if segment == self.snake_head:
                screen.blit(self.head_skin, segment)
            else:
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
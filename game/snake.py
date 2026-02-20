import pygame as pg
from collections import deque
from constants import CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, SNAKE_HEAD_COLOR
from enums import Direction

class Snake:
    # Manages body segments, movement, growth, and self-collision checks.
    def __init__(self):
        start = (
            (SCREEN_WIDTH // CELL_SIZE // 2) * CELL_SIZE,
            (SCREEN_HEIGHT // CELL_SIZE // 2) * CELL_SIZE,
        )
        self.snake = deque([start])          # ordered body (head at index 0)
        self.snake_positions = {start}       # O(1) occupancy checks
        self.snake_head = start

        self.skin = pg.Surface((CELL_SIZE, CELL_SIZE))
        self.skin.fill((255, 255, 255))
        self.head_skin = pg.Surface((CELL_SIZE, CELL_SIZE))
        self.head_skin.fill(SNAKE_HEAD_COLOR)

        self.pending_growth = 0

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

        if self.pending_growth > 0:
            self.pending_growth -= 1
        else:
            tail = self.snake.pop()
            self.snake_positions.remove(tail)

        self.snake.appendleft(new_head)
        self.snake_positions.add(new_head)
        self.snake_head = new_head

    def grow(self):
        self.pending_growth += 1
import pygame as pg
from collections import deque
from constants import CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, SNAKE_HEAD_COLOR
from enums import Direction

class Snake:
    def __init__(self):
        start = (
            SCREEN_WIDTH // CELL_SIZE // 2,
            SCREEN_HEIGHT // CELL_SIZE // 2,
        )
        self.snake = deque([start])
        self.snake_positions = {start}
        self.snake_head = start
        self.direction = Direction.RIGHT

        self.skin = pg.Surface((CELL_SIZE, CELL_SIZE))
        self.skin.fill((255, 255, 255))
        self.head_skin = pg.Surface((CELL_SIZE, CELL_SIZE))
        self.head_skin.fill(SNAKE_HEAD_COLOR)

        self.pending_growth = 0

    def draw_snake(self, screen):
        for segment in self.snake:
            pixel_position = (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE)
            if segment == self.snake_head:
                screen.blit(self.head_skin, pixel_position)
            else:
                screen.blit(self.skin, pixel_position)

    def move(self):
        direction_value = self.direction.value if hasattr(self.direction, 'value') else self.direction
        
        if direction_value == Direction.UP.value:
            new_head = (self.snake_head[0], self.snake_head[1] - 1)
        elif direction_value == Direction.DOWN.value:
            new_head = (self.snake_head[0], self.snake_head[1] + 1)
        elif direction_value == Direction.LEFT.value:
            new_head = (self.snake_head[0] - 1, self.snake_head[1])
        elif direction_value == Direction.RIGHT.value:
            new_head = (self.snake_head[0] + 1, self.snake_head[1])
        else:
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
    
    def get_direction(self):
        return self.direction
    
    def set_direction(self, new_direction):
        opposite_directions = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        if new_direction != opposite_directions.get(self.direction, None):
            self.direction = new_direction
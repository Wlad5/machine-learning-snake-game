import pygame as pg
import random as rd
from constants import FOOD_COLOR, CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT

class Food:
    # Holds position and respawn logic (validated against Snake).
    def __init__(self, board, snake):
        self.board = board
        self.snake = snake
        self.color = FOOD_COLOR
        self.position = self.spawn_food()

    def spawn_food(self):
        while True:
            position_x = rd.randint(0, self.board.cols - 1)
            position_y = rd.randint(0, self.board.rows - 1)
            position = (position_x, position_y)
            if position not in self.snake.snake_positions:
                self.position = position
                return self.position

    def draw_food(self, screen):
        pixel_x, pixel_y = self.board.to_pixel(self.position)
        food_rectangle = pg.Rect(pixel_x, pixel_y, CELL_SIZE, CELL_SIZE)
        pg.draw.rect(screen, self.color, food_rectangle)
    
    def delete_food(self):
        self.position = self.spawn_food()
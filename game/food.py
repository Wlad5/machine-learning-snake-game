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
            position_x = rd.randint(0, (self.board.get_width() // CELL_SIZE) - 1) * CELL_SIZE
            position_y = rd.randint(0, (self.board.get_height() // CELL_SIZE) - 1) * CELL_SIZE
            position = (position_x, position_y)
            if position not in self.snake.snake:
                self.position = position
                return self.position

    def draw_food(self, screen):
        food_rectangle = pg.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE)
        pg.draw.rect(screen, self.color, food_rectangle)
    
    def delete_food(self):
        self.position = self.spawn_food()
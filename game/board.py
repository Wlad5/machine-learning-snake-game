import numpy as np
from enums import CellType

class Board:
    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.grid = np.full((self.rows, self.cols), CellType.EMPTY.value, dtype=np.int8)

    def get_width(self):
        return self.cols * self.cell_size
    
    def get_height(self):
        return self.rows * self.cell_size
    
    def get_cell_size(self):
        return self.cell_size

    def to_cell(self, pixel_position):
        return (pixel_position[0] // self.cell_size, pixel_position[1] // self.cell_size)

    def to_pixel(self, cell_position):
        return (cell_position[0] * self.cell_size, cell_position[1] * self.cell_size)

    def in_bounds_cell(self, cell_position):
        x, y = cell_position
        return 0 <= x < self.cols and 0 <= y < self.rows

    def reset_grid(self):
        self.grid.fill(CellType.EMPTY.value)

    def set_cell(self, cell_position, cell_type):
        x, y = cell_position
        if self.in_bounds_cell(cell_position):
            self.grid[y, x] = cell_type.value

    def get_cell(self, cell_position):
        if not self.in_bounds_cell(cell_position):
            return CellType.WALL
        x, y = cell_position
        return CellType(int(self.grid[y, x]))

    def refresh_entities(self, snake_cells, food_cell):
        self.reset_grid()
        for cell in snake_cells:
            self.set_cell(cell, CellType.SNAKE)
        if food_cell is not None and self.get_cell(food_cell) == CellType.EMPTY:
            self.set_cell(food_cell, CellType.FOOD)
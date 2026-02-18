import pygame as pg

class Renderer:
    # Draws game state; keep rendering separate from logic.
    def __init__(self, screen, board):
        self.screen = screen
        self.board = board
    
    def draw_board(self):
        self.screen.fill((0, 0, 0))
        cell_size = self.board.get_cell_size()
        board_width = self.board.get_width()
        board_height = self.board.get_height()

        for x in range(0, board_width, cell_size):
            pg.draw.line(self.screen, (40, 40, 40), (x, 0), (x, board_height))
        
        for y in range(0, board_height, cell_size):
            pg.draw.line(self.screen, (40, 40, 40), (0, y), (board_width, y))
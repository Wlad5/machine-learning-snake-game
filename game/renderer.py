import pygame as pg
from constants import UI_PANEL_HEIGHT
from enums import CellType

class Renderer:
    def __init__(self, screen, board):
        self.screen = screen
        self.board = board
        self.status_font = pg.font.SysFont(None, 28)
    
    def draw_board(self):
        self.screen.fill((0, 0, 0))
        cell_size = self.board.get_cell_size()
        board_width = self.board.get_width()
        board_height = self.board.get_height()

        # Draw grid lines
        for x in range(0, board_width, cell_size):
            pg.draw.line(self.screen, (40, 40, 40), (x, 0), (x, board_height))
        
        for y in range(0, board_height, cell_size):
            pg.draw.line(self.screen, (40, 40, 40), (0, y), (board_width, y))
    
    def draw_game_status_panel(self, game_status, food_count, movement_direction):
        board_width = self.board.get_width()
        board_height = self.board.get_height()

        panel_rect = pg.Rect(0, board_height, board_width, UI_PANEL_HEIGHT)
        pg.draw.rect(self.screen, (50, 50, 50), panel_rect)

        status_text = self.status_font.render(f"Status: {game_status.name}", True, (255, 255, 255))
        self.screen.blit(status_text, (10, board_height + 10))

        counter_text = self.status_font.render(f"Food eaten: {food_count}", True, (255, 255, 255))
        self.screen.blit(counter_text, (10, board_height + 50))
        direction_text = self.status_font.render(f"Direction: {movement_direction.name}", True, (255, 255, 255))
        self.screen.blit(direction_text, (10, board_height + 70))
import pygame as pg
from constants import UI_PANEL_HEIGHT

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

    def draw_local_grid(self, snake, board):
        """Overlay the 3x3 local grid vision around the snake head."""
        cell_size = self.board.get_cell_size()
        head_x, head_y = snake.snake_head
        food_positions = set()  # populated by draw_local_grid_with_food

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = head_x + dx, head_y + dy
                px, py = nx * cell_size, ny * cell_size

                is_obstacle = not board.in_bounds_cell((nx, ny)) or (nx, ny) in snake.snake_positions

                overlay = pg.Surface((cell_size, cell_size), pg.SRCALPHA)
                if is_obstacle:
                    overlay.fill((220, 50, 50, 110))   # red – obstacle
                    border_color = (220, 80, 80)
                else:
                    overlay.fill((50, 160, 220, 60))   # blue – free cell
                    border_color = (80, 140, 200)

                self.screen.blit(overlay, (px, py))
                pg.draw.rect(self.screen, border_color, (px, py, cell_size, cell_size), 1)

    def draw_local_grid_with_food(self, snake, board, food):
        """Overlay the 3x3 local grid vision, differentiating food cells."""
        cell_size = self.board.get_cell_size()
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = head_x + dx, head_y + dy
                px, py = nx * cell_size, ny * cell_size

                is_obstacle = not board.in_bounds_cell((nx, ny)) or (nx, ny) in snake.snake_positions
                is_food = (nx, ny) == (food_x, food_y)

                overlay = pg.Surface((cell_size, cell_size), pg.SRCALPHA)
                if is_obstacle:
                    overlay.fill((220, 50, 50, 110))   # red – obstacle
                    border_color = (220, 80, 80)
                elif is_food:
                    overlay.fill((80, 220, 80, 110))   # green – food
                    border_color = (80, 220, 80)
                else:
                    overlay.fill((50, 160, 220, 60))   # blue – free cell
                    border_color = (80, 140, 200)

                self.screen.blit(overlay, (px, py))
                pg.draw.rect(self.screen, border_color, (px, py, cell_size, cell_size), 1)

    def draw_rays(self, snake, board, food):
        """Draw the 8 raycasting rays from the snake head."""
        cell_size = self.board.get_cell_size()
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position

        head_px = head_x * cell_size + cell_size // 2
        head_py = head_y * cell_size + cell_size // 2

        ray_directions = [
            (0, -1),   # up
            (0,  1),   # down
            (-1, 0),   # left
            (1,  0),   # right
            (-1, -1),  # up-left
            (1,  -1),  # up-right
            (-1,  1),  # down-left
            (1,   1),  # down-right
        ]

        max_distance = max(board.cols, board.rows)

        for dx, dy in ray_directions:
            # Walk the ray to find the last in-bounds cell before an obstacle
            end_x, end_y = head_x, head_y
            hit_obstacle = False
            for i in range(1, max_distance + 1):
                nx, ny = head_x + dx * i, head_y + dy * i
                if not board.in_bounds_cell((nx, ny)):
                    hit_obstacle = True
                    break
                if (nx, ny) in snake.snake_positions:
                    end_x, end_y = nx, ny
                    hit_obstacle = True
                    break
                end_x, end_y = nx, ny

            end_px = end_x * cell_size + cell_size // 2
            end_py = end_y * cell_size + cell_size // 2

            # Check if food lies on this ray
            food_in_ray = self._food_on_ray(food_x, food_y, head_x, head_y, dx, dy)

            ray_color = (255, 230, 50) if food_in_ray else (80, 160, 255)
            pg.draw.line(self.screen, ray_color, (head_px, head_py), (end_px, end_py), 1)

            hit_color = (255, 80, 80) if hit_obstacle else (80, 255, 80)
            pg.draw.circle(self.screen, hit_color, (end_px, end_py), 3)

    @staticmethod
    def _food_on_ray(food_x, food_y, head_x, head_y, dx, dy):
        """Return True if food lies exactly on the ray from head in direction (dx, dy)."""
        if dx == 0 and dy == 0:
            return False
        if dx != 0:
            t = (food_x - head_x) / dx
            if t > 0 and (food_y - head_y) == dy * t:
                return True
        if dy != 0:
            t = (food_y - head_y) / dy
            if t > 0 and (food_x - head_x) == dx * t:
                return True
        return False
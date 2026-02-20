import pygame as pg
from board import Board
from snake import Snake
from food import Food
from collision_rules import CollisionRules
from input_handler import InputHandler
from renderer import Renderer
from enums import Direction, GameStatus
from constants import CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, UI_PANEL_HEIGHT

class Game:
    # Owns the loop, timing, 
    # input dispatch, and high‑level state (running, paused, game over).
    def __init__(self):
        pg.init()
        self.board = Board(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.screen = pg.display.set_mode(
            (self.board.get_width(), self.board.get_height() + UI_PANEL_HEIGHT)
        )
        pg.display.set_caption("Snake Game")

        self.renderer = Renderer(self.screen, self.board)
        self.input_handler = InputHandler()
        self.snake = Snake()
        self.food = Food(self.board, self.snake)

        self.status = GameStatus.RUNNING
        self.direction = Direction.RIGHT
        self.food_eaten = 0

        self.clock = pg.time.Clock()
        self.move_timer_ms = 0
        self.move_delay_ms = 420

    def run(self):
        running = True
        while running and self.status not in (GameStatus.GAME_OVER, GameStatus.WIN):
            dt_ms = self.clock.tick(60)
            running = self._process_events()
            self._update(dt_ms)
            self._render()
        pg.quit()

    def _process_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False

            new_direction = self.input_handler.handle_event(event)
            if new_direction:
                self.direction = new_direction
        return True

    def _update(self, dt_ms):
        if self.status != GameStatus.RUNNING:
            return

        self.move_timer_ms += dt_ms
        while self.move_timer_ms >= self.move_delay_ms:
            self.snake.move(self.direction)
            self.move_timer_ms -= self.move_delay_ms
            self._resolve_collisions()

    def _resolve_collisions(self):
        food_event = CollisionRules.check_food_collision(self.food, self.snake)
        if food_event:
            self.food_eaten += 1

        game_status = CollisionRules.check_game_over(self.snake, self.board)
        if game_status in (GameStatus.GAME_OVER, GameStatus.WIN):
            self.status = game_status

    def _render(self):
        self.renderer.draw_board()
        self.food.draw_food(self.screen)
        self.snake.draw_snake(self.screen)
        self.renderer.draw_game_status_panel(
            self.status, self.food_eaten, self.direction
        )
        pg.display.flip()

if __name__ == "__main__":
    game = Game()
    game.run()

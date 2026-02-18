import pygame as pg
from board import Board
from snake import Snake
from food import Food
from collision_rules import CollisionRules
from input_handler import InputHandler
from renderer import Renderer
from enums import Direction, GameStatus
from constants import CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT

class Game:
    # Owns the loop, timing, 
    # input dispatch, and high‑level state (running, paused, game over).
    def __init__(self):
        pg.init()
        self.board = Board(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.screen = pg.display.set_mode((self.board.get_width(), self.board.get_height()))
        self.status = GameStatus.PAUSED
        self.renderer = Renderer(self.screen, self.board)
        self.snake = Snake()
        self.input_handler = InputHandler()
        self.food = Food(self.board, self.snake)
        pg.display.set_caption("Snake Game")

    def run(self):
        running = True
        self.status = GameStatus.RUNNING
        clock = pg.time.Clock()
        direction = Direction.RIGHT
        move_timer = 0
        move_delay = 60 #ms
        while running and self.status == GameStatus.RUNNING:
            dt = clock.tick(60)
            move_timer += dt
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                new_direction = self.input_handler.handle_event(event)
                if new_direction:
                    direction = new_direction
            if move_timer >= move_delay:
                self.snake.move(direction)
                move_timer = 0
            self.renderer.draw_board()
            self.food.draw_food(self.screen)
            self.snake.draw_snake(self.screen)
            CollisionRules.check_food_collision(self.food, self.snake)
            game_status = CollisionRules.check_game_over(self.snake, self.board)
            if game_status == GameStatus.GAME_OVER:
                self.status = GameStatus.GAME_OVER

            pg.display.flip()
        pg.quit()

class GameState:
    # Snapshot of current state for ML (snake, food, score, etc.).
    pass

if __name__ == "__main__":
    game = Game()
    game.run()

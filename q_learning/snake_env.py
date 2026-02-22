import pygame as pg
from game import (Board,
                  Snake,
                  Food,
                  Game,
                  Renderer,
                  UI_PANEL_HEIGHT,
                  SCREEN_WIDTH,
                  SCREEN_HEIGHT,
                  CELL_SIZE,
                  Direction
                  )

class Snake_Env:
    def __init__(self,
                 render_mode=True,
                 max_steps_per_episode=1000,
                 food_reward=10,
                 death_penalty=-10,
                 per_step_reward=-0.1):
        pg.init()
        self.render_mode = render_mode
        self.max_steps_per_episode = max_steps_per_episode
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.per_step_reward = per_step_reward

        self.board = Board(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.snake = Snake()
        self.food = Food(self.board, self.snake)
        self.game = Game(self.board, self.snake, self.food)
        
        self.status = Game.get_status(self.game)
        self.step_count = 0
        self.score = 0

        self.screen = pg.display.set_mode(self.board.get_width(), self.board.get_height() + UI_PANEL_HEIGHT)
        pg.display.set_caption("Snake Game")
        self.clock = pg.time.Clock()
        self.renderer = Renderer(self.screen, self.board)

    def reset():
        pass

    def step():
        pass

    def get_state(self):
        state = []
        state.append(int(self.snake.get_direction() == Direction.UP))
        state.append(int(self.snake.get_direction() == Direction.DOWN))
        state.append(int(self.snake.get_direction() == Direction.LEFT))
        state.append(int(self.snake.get_direction() == Direction.RIGHT))

    def render():
        pass
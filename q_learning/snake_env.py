import pygame as pg
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from state_encoding import State_Encoding
from game.board import Board
from game.enums import *
from game.snake import Snake
from game.food import Food
from game.renderer import Renderer
from game.constants import *


class Snake_Env:
    def __init__(
        self,
        render_mode=False,
        max_steps_per_episode=1000,
        food_reward=10,
        death_penalty=-10,
        per_step_reward=-0.1,
        reward_for_winning=1000,
    ):
        pg.init()
        self.render_mode = render_mode
        self.max_steps_per_episode = max_steps_per_episode
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.per_step_reward = per_step_reward
        self.reward_for_winning = reward_for_winning

        self.board = Board(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.snake = Snake()
        self.food = Food(self.board, self.snake)
        self.state_encoder = State_Encoding()

        self.game_status = GameStatus.RUNNING
        self.done = False
        self.step_count = 0
        self.score = 0

        self.screen = None
        self.clock = None
        self.renderer = None

        if self.render_mode:
            self.screen = pg.display.set_mode(
                (self.board.get_width(), self.board.get_height() + UI_PANEL_HEIGHT)
            )
            pg.display.set_caption("Snake Game")
            self.clock = pg.time.Clock()
            self.renderer = Renderer(self.screen, self.board)

    def reset(self):
        self.snake = Snake()
        self.food = Food(self.board, self.snake)
        
        # Ensure the snake has Direction.RIGHT set initially
        self.snake.set_direction(Direction.RIGHT)
        
        # Refresh board with initial positions
        self.board.refresh_entities(self.snake.snake, self.food.position)

        self.step_count = 0
        self.score = 0
        self.done = False

        return self.get_state()

    def step(self, action):
        if self.done:
            info = {"score": self.score, "steps": self.step_count}
            return self.get_state(), 0.0, True, info

        reward = float(self.per_step_reward)

        action_to_direction = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT,
        }

        new_direction = action_to_direction.get(action, self.snake.get_direction())
        self.snake.set_direction(new_direction)

        self.snake.move()
        self.step_count += 1

        head = self.snake.snake_head
        body_without_head = list(self.snake.snake)[1:]

        hit_wall = not self.board.in_bounds_cell(head)
        hit_self = head in body_without_head

        if hit_wall or hit_self:
            reward += self.death_penalty
            self.done = True
        elif head == self.food.position:
            self.score += 1
            reward += self.food_reward
            self.snake.grow()
            length_reward = len(self.snake.snake) * 10
            reward += length_reward
    
            total_cells = self.board.cols * self.board.rows
            if len(self.snake.snake) >= total_cells:
                reward += self.reward_for_winning
            else:
                self.food.delete_food()

        if self.step_count >= self.max_steps_per_episode:
            self.done = True

        next_state = self.get_state()
        info = {"score": self.score, "steps": self.step_count}

        return next_state, reward, self.done, info

    def get_state(self):
        return self.state_encoder.encode(self.board, self.snake, self.food)

    def render(self, fps=30):
        if not self.render_mode:
            return

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

        self.board.refresh_entities(self.snake.snake, self.food.position)
        self.renderer.draw_board()
        self.renderer.draw_game_status_panel(self.game_status, self.score, self.snake.get_direction())
        self.snake.draw_snake(self.screen)
        self.food.draw_food(self.screen)
        pg.display.flip()

        if self.clock is not None:
            self.clock.tick(fps)

    def close(self):
        if self.render_mode:
            pg.quit()
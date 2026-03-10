import pygame as pg
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from linear_q_learning_state_encoding import LinearQLearningStateEncoding
from game.board import Board
from game.enums import *
from game.snake import Snake
from game.food import Food
from game.renderer import Renderer
from game.constants import *


class LinearQLearningEnvironment:
    """
    Environment for training a Linear Q-Learning agent on the Snake game.
    Provides game mechanics and rendering capabilities.
    """
    
    def __init__(
        self,
        render_mode=False,
        max_steps_per_episode=1000,
        food_reward=10,
        death_penalty=-10,
        per_step_reward=-0.1,
        reward_for_winning=1000,
    ):
        """
        Initialize the Linear Q-Learning environment.
        
        Args:
            render_mode: Whether to render the game visually during training
            max_steps_per_episode: Maximum steps allowed per episode
            food_reward: Reward for eating food
            death_penalty: Penalty for dying
            per_step_reward: Reward/penalty for each step
            reward_for_winning: Reward for filling the entire board
        """
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
        self.state_encoder = LinearQLearningStateEncoding()

        self.game_status = GameStatus.RUNNING
        self.done = False
        self.step_count = 0
        self.score = 0

        self.screen = None
        self.clock = None
        self.renderer = None

        if self.render_mode:
            self.screen = pg.display.set_mode(
                (WINDOW_HEIGHT, WINDOW_WIDTH)
            )
            pg.display.set_caption("Linear Q-Learning Snake")
            self.clock = pg.time.Clock()
            self.renderer = Renderer(self.screen, self.board)

    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            tuple: Initial state (feature vector)
        """
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
        """
        Execute one step of the environment with the given action.
        
        Args:
            action: Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
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
        """
        Get the encoded state of the current game.
        
        Returns:
            tuple: State feature vector
        """
        return self.state_encoder.encode(self.board, self.snake, self.food)

    def render(self, fps=30):
        """
        Render the current game state to the screen.
        
        Args:
            fps: Frames per second for rendering
        """
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
        """Close the environment and pygame."""
        if self.render_mode:
            pg.quit()
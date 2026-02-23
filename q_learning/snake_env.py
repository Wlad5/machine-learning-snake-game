import pygame as pg
from game import (
    Board,
    Snake,
    Food,
    Renderer,
    UI_PANEL_HEIGHT,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    CELL_SIZE,
    Direction,
)


class Snake_Env:
    def __init__(
        self,
        render_mode=True,
        max_steps_per_episode=1000,
        food_reward=10,
        death_penalty=-10,
        per_step_reward=-0.1,
    ):
        pg.init()
        self.render_mode = render_mode
        self.max_steps_per_episode = max_steps_per_episode
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.per_step_reward = per_step_reward

        self.board = Board(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.snake = Snake()
        self.food = Food(self.board, self.snake)

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
            self.food.delete_food()

        if self.step_count >= self.max_steps_per_episode:
            self.done = True

        next_state = self.get_state()
        info = {"score": self.score, "steps": self.step_count}

        return next_state, reward, self.done, info

    def get_state(self):
        head_x, head_y = self.snake.snake_head
        food_x, food_y = self.food.position
        snake_direction = self.snake.get_direction()

        def is_danger(cell_x, cell_y):
            next_cell = (cell_x, cell_y)
            return (
                (not self.board.in_bounds_cell(next_cell))
                or (next_cell in self.snake.snake_positions)
            )

        state = (
            int(snake_direction == Direction.UP),
            int(snake_direction == Direction.DOWN),
            int(snake_direction == Direction.LEFT),
            int(snake_direction == Direction.RIGHT),
            int(food_y < head_y),  # food up
            int(food_y > head_y),  # food down
            int(food_x < head_x),  # food left
            int(food_x > head_x),  # food right
            int(is_danger(head_x, head_y - 1)),  # danger up
            int(is_danger(head_x, head_y + 1)),  # danger down
            int(is_danger(head_x - 1, head_y)),  # danger left
            int(is_danger(head_x + 1, head_y)),  # danger right
        )
        return state

    def render(self, fps=30):
        if not self.render_mode:
            return

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

        self.board.refresh_entities(self.snake.snake, self.food.position)
        self.renderer.draw_board()
        self.snake.draw_snake(self.screen)
        self.food.draw_food(self.screen)
        pg.display.flip()

        if self.clock is not None:
            self.clock.tick(fps)

    def close(self):
        if self.render_mode:
            pg.quit()
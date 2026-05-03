import numpy as np
from game.enums import Direction


class DQNDistanceStateEncoding:
    """State Representation 1: Relative Distance Encoding
    Encodes normalized distances to obstacles and food (17 features)

    Original 13 features + 4 immediate danger flags (wall OR body).
    The original encoding only tracked wall distances, making the snake
    completely blind to its own body segments.  The 4 danger flags close
    that gap and are grid-size invariant (they do not depend on board dims).
    """

    def encode(self, board, snake, food):
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position
        snake_direction = snake.get_direction()

        direction_value = snake_direction.value if hasattr(snake_direction, 'value') else snake_direction

        # Manhattan distances normalized by board size
        max_distance = board.cols + board.rows
        food_distance = (abs(food_x - head_x) + abs(food_y - head_y)) / max_distance

        # Distance to walls in 4 directions (wall-only, as before)
        distance_up = (head_y) / board.rows
        distance_down = (board.rows - head_y - 1) / board.rows
        distance_left = (head_x) / board.cols
        distance_right = (board.cols - head_x - 1) / board.cols

        state = np.array([
            int(direction_value == Direction.UP.value),
            int(direction_value == Direction.DOWN.value),
            int(direction_value == Direction.LEFT.value),
            int(direction_value == Direction.RIGHT.value),
            int(food_y < head_y),  # food up
            int(food_y > head_y),  # food down
            int(food_x < head_x),  # food left
            int(food_x > head_x),  # food right
            distance_up,
            distance_down,
            distance_left,
            distance_right,
            food_distance,
            # Immediate danger flags (wall OR body) — 4 new features.
            # These are the critical addition: the previous encoding was blind
            # to body obstacles, which dominates on larger/longer snakes.
            int(self._is_danger(board, snake, head_x, head_y - 1)),  # danger up
            int(self._is_danger(board, snake, head_x, head_y + 1)),  # danger down
            int(self._is_danger(board, snake, head_x - 1, head_y)),  # danger left
            int(self._is_danger(board, snake, head_x + 1, head_y)),  # danger right
        ], dtype=np.float32)
        return state

    def _is_danger(self, board, snake, cell_x, cell_y):
        next_cell = (cell_x, cell_y)
        return (
            (not board.in_bounds_cell(next_cell))
            or (next_cell in snake.snake_positions)
        )

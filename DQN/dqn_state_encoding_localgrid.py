import numpy as np
from game.enums import Direction


class DQNLocalGridStateEncoding:
    """State Representation 3: Local Grid Vision
    5x5 grid around snake head with occupancy markers (61 features)

    61 = 4 direction + 4 binary food dir + 48 local grid cells + 2 norm food offset + 1 norm food distance + tail offset (2).

    On 5x5+ grids the agent must navigate across several cells, so binary
    direction alone is ambiguous — a food 1 square away and 5 squares away look
    identical.  Adding signed normalized food offsets and a scalar Manhattan
    distance gives the agent the magnitude it is missing.  The tail offset further
    helps avoid self-trapping as the snake grows longer on the larger boards.
    """
    
    def encode(self, board, snake, food):
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position
        snake_direction = snake.get_direction()
        
        direction_value = snake_direction.value if hasattr(snake_direction, 'value') else snake_direction

        # Signed normalized food offsets (scale-invariant).
        norm_food_dx = (food_x - head_x) / max(board.cols, 1)
        norm_food_dy = (food_y - head_y) / max(board.rows, 1)

        # Normalized Manhattan food distance.
        max_distance = board.cols + board.rows
        norm_food_dist = (abs(food_x - head_x) + abs(food_y - head_y)) / max_distance

        # Normalized tail offset.
        snake_body = list(snake.snake)
        tail_x, tail_y = snake_body[-1] if len(snake_body) > 1 else (head_x, head_y)
        max_dim = max(board.cols, board.rows)
        norm_tail_dx = (tail_x - head_x) / max_dim
        norm_tail_dy = (tail_y - head_y) / max_dim

        # 5x5 grid around head (24 cells, centre is head and skipped).
        grid_state = []
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    continue  # Skip center (head position)
                nx, ny = head_x + dx, head_y + dy
                
                is_obstacle = not board.in_bounds_cell((nx, ny)) or (nx, ny) in snake.snake_positions
                is_food = (nx, ny) == (food_x, food_y)
                
                grid_state.append(int(is_obstacle))
                grid_state.append(int(is_food))
        
        state = np.array([
            int(direction_value == Direction.UP.value),
            int(direction_value == Direction.DOWN.value),
            int(direction_value == Direction.LEFT.value),
            int(direction_value == Direction.RIGHT.value),
            int(food_x < head_x),  # food general directions (binary)
            int(food_x > head_x),
            int(food_y < head_y),
            int(food_y > head_y),
        ] + grid_state + [
            norm_food_dx,    # signed normalized horizontal offset
            norm_food_dy,    # signed normalized vertical offset
            norm_food_dist,  # scalar distance — distinguishes near vs far food
            norm_tail_dx,    # tail direction for self-trap avoidance
            norm_tail_dy,
        ], dtype=np.float32)
        
        return state

import numpy as np
from game.enums import Direction


class DQNBodyAwarenessStateEncoding:
    """State Representation 4: Snake Length & Body Proximity
    Includes snake body information and proximity warnings (14 features)"""
    
    def encode(self, board, snake, food):
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position
        snake_direction = snake.get_direction()
        
        direction_value = snake_direction.value if hasattr(snake_direction, 'value') else snake_direction
        
        # Normalize snake length
        max_length = board.cols * board.rows
        normalized_length = len(snake.snake_positions) / max_length
        
        # Check if body is adjacent (1 move away)
        body_nearby = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = head_x + dx, head_y + dy
            # Exclude head position
            if (nx, ny) in snake.snake_positions and (nx, ny) != (head_x, head_y):
                body_nearby += 1
        
        state = np.array([
            int(direction_value == Direction.UP.value),
            int(direction_value == Direction.DOWN.value),
            int(direction_value == Direction.LEFT.value),
            int(direction_value == Direction.RIGHT.value),
            int(food_y < head_y),  # food up
            int(food_y > head_y),  # food down
            int(food_x < head_x),  # food left
            int(food_x > head_x),  # food right
            int(self._is_danger(board, snake, head_x, head_y - 1)),  # danger up
            int(self._is_danger(board, snake, head_x, head_y + 1)),  # danger down
            int(self._is_danger(board, snake, head_x - 1, head_y)),  # danger left
            int(self._is_danger(board, snake, head_x + 1, head_y)),  # danger right
            normalized_length,
            body_nearby / 4.0,  # Normalized by max 4 adjacent cells
        ], dtype=np.float32)
        return state
    
    def _is_danger(self, board, snake, cell_x, cell_y):
        next_cell = (cell_x, cell_y)
        return (
            (not board.in_bounds_cell(next_cell))
            or (next_cell in snake.snake_positions)
        )

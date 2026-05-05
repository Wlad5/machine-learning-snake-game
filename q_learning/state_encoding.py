from game.enums import Direction

class State_Encoding:
    def encode(self, board, snake, food):
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position
        snake_direction = snake.get_direction()
        
        direction_value = snake_direction.value if hasattr(snake_direction, 'value') else snake_direction

        norm_food_dx = (food_x - head_x) / max(board.cols, 1)
        norm_food_dy = (food_y - head_y) / max(board.rows, 1)

        snake_body = list(snake.snake)
        tail_x, tail_y = snake_body[-1] if len(snake_body) > 1 else (head_x, head_y)
        max_dim = max(board.cols, board.rows)
        norm_tail_dx = (tail_x - head_x) / max_dim
        norm_tail_dy = (tail_y - head_y) / max_dim

        state = (
            int(direction_value == Direction.UP.value),       # direction UP
            int(direction_value == Direction.DOWN.value),     # direction DOWN
            int(direction_value == Direction.LEFT.value),     # direction LEFT
            int(direction_value == Direction.RIGHT.value),    # direction RIGHT
            norm_food_dx,   # signed normalized horizontal offset to food
            norm_food_dy,   # signed normalized vertical offset to food
            int(food_y < head_y),  # food up (binary)
            int(food_y > head_y),  # food down
            int(food_x < head_x),  # food left
            int(food_x > head_x),  # food right
            int(self._is_danger(board, snake, head_x, head_y - 1)),  # danger up
            int(self._is_danger(board, snake, head_x, head_y + 1)),  # danger down
            int(self._is_danger(board, snake, head_x - 1, head_y)),  # danger left
            int(self._is_danger(board, snake, head_x + 1, head_y)),  # danger right
            norm_tail_dx,   # signed normalized horizontal offset to tail
            norm_tail_dy,   # signed normalized vertical offset to tail
        )
        return state
    
    def _is_danger(self, board, snake, cell_x, cell_y):
            next_cell = (cell_x, cell_y)
            return (
                (not board.in_bounds_cell(next_cell))
                or (next_cell in snake.snake_positions)
            )
from game.enums import Direction

class State_Encoding:
    def encode(self, board, snake, food):
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position
        snake_direction = snake.get_direction()
        
        # Use both == and .value comparison for robustness
        direction_value = snake_direction.value if hasattr(snake_direction, 'value') else snake_direction
        
        state = (
            int(direction_value == Direction.UP.value),       # direction UP
            int(direction_value == Direction.DOWN.value),     # direction DOWN
            int(direction_value == Direction.LEFT.value),     # direction LEFT
            int(direction_value == Direction.RIGHT.value),    # direction RIGHT
            int(food_y < head_y),  # food up
            int(food_y > head_y),  # food down
            int(food_x < head_x),  # food left
            int(food_x > head_x),  # food right
            int(self._is_danger(board, snake, head_x, head_y - 1)),  # danger up
            int(self._is_danger(board, snake, head_x, head_y + 1)),  # danger down
            int(self._is_danger(board, snake, head_x - 1, head_y)),  # danger left
            int(self._is_danger(board, snake, head_x + 1, head_y)),  # danger right
        )
        return state
    
    def _is_danger(self, board, snake, cell_x, cell_y):
            next_cell = (cell_x, cell_y)
            return (
                (not board.in_bounds_cell(next_cell))
                or (next_cell in snake.snake_positions)
            )
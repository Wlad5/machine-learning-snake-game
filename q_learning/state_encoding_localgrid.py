from game.enums import Direction


class LocalGridStateEncoding:
    """State Representation 3: Local Grid Vision
    3x3 grid around snake head with occupancy markers (16 features)"""
    
    def encode(self, board, snake, food):
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position
        snake_direction = snake.get_direction()
        
        direction_value = snake_direction.value if hasattr(snake_direction, 'value') else snake_direction
        
        # 3x3 grid around head
        grid_state = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip center (head position)
                nx, ny = head_x + dx, head_y + dy
                
                is_obstacle = not board.in_bounds_cell((nx, ny)) or (nx, ny) in snake.snake_positions
                is_food = (nx, ny) == (food_x, food_y)
                
                grid_state.append(int(is_obstacle))
                grid_state.append(int(is_food))
        
        state = tuple([
            int(direction_value == Direction.UP.value),
            int(direction_value == Direction.DOWN.value),
            int(direction_value == Direction.LEFT.value),
            int(direction_value == Direction.RIGHT.value),
            int(food_x < head_x),  # food general directions
            int(food_x > head_x),
            int(food_y < head_y),
            int(food_y > head_y),
        ] + grid_state)
        
        return state

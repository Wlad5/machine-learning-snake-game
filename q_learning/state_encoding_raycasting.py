from game.enums import Direction


class RayCastingStateEncoding:
    """State Representation 2: Ray-Casting Vision
    Multiple rays emanating from snake head detecting distance to obstacles/food (20 features)"""
    
    def encode(self, board, snake, food):
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position
        snake_direction = snake.get_direction()
        
        direction_value = snake_direction.value if hasattr(snake_direction, 'value') else snake_direction
        
        # 8 rays: up, down, left, right, and 4 diagonals
        rays = [
            (0, -1),   # up
            (0, 1),    # down
            (-1, 0),   # left
            (1, 0),    # right
            (-1, -1),  # up-left
            (1, -1),   # up-right
            (-1, 1),   # down-left
            (1, 1),    # down-right
        ]
        
        ray_data = []
        for dx, dy in rays:
            distance_to_obstacle = self._cast_ray(board, snake, head_x, head_y, dx, dy)
            food_in_ray = int(self._is_food_in_ray(food_x, food_y, head_x, head_y, dx, dy))
            ray_data.extend([distance_to_obstacle, food_in_ray])
        
        state = tuple([
            int(direction_value == Direction.UP.value),
            int(direction_value == Direction.DOWN.value),
            int(direction_value == Direction.LEFT.value),
            int(direction_value == Direction.RIGHT.value),
        ] + ray_data)
        
        return state
    
    def _cast_ray(self, board, snake, x, y, dx, dy, max_distance=10):
        """Cast a ray from (x, y) in direction (dx, dy) and return normalized distance to obstacle"""
        distance = 0
        for i in range(1, max_distance + 1):
            nx, ny = x + dx * i, y + dy * i
            if not board.in_bounds_cell((nx, ny)) or (nx, ny) in snake.snake_positions:
                return i / max_distance
        return 1.0
    
    def _is_food_in_ray(self, food_x, food_y, head_x, head_y, dx, dy):
        """Check if food is in the direction of the ray"""
        if dx == 0 and dy == 0:
            return False
        if dx != 0:
            t = (food_x - head_x) / dx if dx != 0 else float('inf')
            if t > 0 and (food_y - head_y) == dy * t:
                return True
        if dy != 0:
            t = (food_y - head_y) / dy if dy != 0 else float('inf')
            if t > 0 and (food_x - head_x) == dx * t:
                return True
        return False

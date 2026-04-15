from game.enums import Direction


class LinearQLearningDistanceEncoding:
    """State Representation 1: Relative Distance Encoding (13 features)
    Encodes normalized distances to obstacles and food"""
    
    def encode(self, board, snake, food):
        head_x, head_y = snake.snake_head
        food_x, food_y = food.position
        snake_direction = snake.get_direction()
        
        direction_value = snake_direction.value if hasattr(snake_direction, 'value') else snake_direction
        
        # Manhattan distances normalized by board size
        max_distance = board.cols + board.rows
        food_distance = (abs(food_x - head_x) + abs(food_y - head_y)) / max_distance
        
        # Distance to walls in 4 directions
        distance_up = (head_y) / board.rows
        distance_down = (board.rows - head_y - 1) / board.rows
        distance_left = (head_x) / board.cols
        distance_right = (board.cols - head_x - 1) / board.cols
        
        state = (
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
        )
        return state

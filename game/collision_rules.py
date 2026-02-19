from enums import GameStatus, EventType
from constants import CELL_SIZE

class CollisionRules:
    def check_food_collision(food, snake):
        if snake.snake_head == food.position:
            snake.grow()
            food.delete_food()
            print("Ate food")
            return EventType.ATE_FOOD
        return None
    
    def check_wall_collision(snake, board):
        TOP_WALL = snake.snake_head[1] < 0
        BOTTOM_WALL = snake.snake_head[1] >= board.get_height()
        LEFT_WALL = snake.snake_head[0] < 0
        RIGHT_WALL = snake.snake_head[0] >= board.get_width()

        if TOP_WALL or BOTTOM_WALL or LEFT_WALL or RIGHT_WALL:
            print("Hit wall")
            return EventType.HIT_WALL
        return None

    def check_self_collision(snake):
        # Only check for self-collision if snake is longer than 2 tiles
        if len(snake.snake) > 2:
            for segment in snake.snake[1:]:
                if snake.snake_head == segment:
                    print("Hit self")
                    return EventType.HIT_SELF
        return None
    
    def check_win(snake, board):
        total_cells = (board.get_width() // CELL_SIZE) * (board.get_height() // CELL_SIZE)
        if len(snake.snake) >= total_cells:
            print("Won")
            return GameStatus.WIN
    
    def check_game_over(snake, board):
        if CollisionRules.check_win(snake, board):
            return GameStatus.WIN
        
        if (CollisionRules.check_wall_collision(snake, board) 
            or CollisionRules.check_self_collision(snake)):
            return GameStatus.GAME_OVER
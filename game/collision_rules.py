from enums import GameStatus, EventType

class CollisionRules:
    def check_food_collision(food, snake):
        if snake.snake_head == food.position:
            snake.grow()
            food.delete_food()
            print("Ate food")
            return EventType.ATE_FOOD
        return None
    
    def check_wall_collision(snake, board):
        if not board.in_bounds_cell(snake.snake_head):
            print("Hit wall")
            return EventType.HIT_WALL
        return None

    def check_self_collision(snake):
        if len(snake.snake_positions) != len(snake.snake):
            print("Hit self")
            return EventType.HIT_SELF
        return None
    
    def check_win(snake, board):
        total_cells = board.cols * board.rows
        if len(snake.snake) >= total_cells:
            print("Won")
            return GameStatus.WIN
    
    def check_game_over(snake, board):
        if CollisionRules.check_win(snake, board):
            return GameStatus.WIN
        
        if (CollisionRules.check_wall_collision(snake, board) 
            or CollisionRules.check_self_collision(snake)):
            return GameStatus.GAME_OVER
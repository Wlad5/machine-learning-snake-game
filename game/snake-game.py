from enum import Enum
import pygame as pg
from board import Board
import random as rd

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Action(Enum):
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STRAIGHT = 3
    MOVE_UP = 4
    MOVE_DOWN = 5

class GameStatus(Enum):
    RUNNING = 1
    PAUSED = 2
    GAME_OVER = 3
    WIN = 4

class CellType(Enum):
    EMPTY = 0
    SNAKE = 1
    FOOD = 3
    WALL = 4

class EventType(Enum):
    ATE_FOOD = 1
    HIT_WALL = 2
    HIT_SELF = 3

FOOD_COLOR = (255, 0, 0)
CELL_SIZE = 10
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200


class Game:
    # Owns the loop, timing, 
    # input dispatch, and high‑level state (running, paused, game over).
    def __init__(self):
        pg.init()
        self.board = Board(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.screen = pg.display.set_mode((self.board.get_width(), self.board.get_height()))
        self.status = GameStatus.PAUSED
        self.renderer = Renderer(self.screen, self.board)
        self.snake = Snake()
        self.input_handler = InputHandler()
        self.food = Food(self.board, self.snake)
        pg.display.set_caption("Snake Game")

    def run(self):
        running = True
        self.status = GameStatus.RUNNING
        clock = pg.time.Clock()
        direction = Direction.RIGHT
        move_timer = 0
        move_delay = 60 #ms
        while running and self.status == GameStatus.RUNNING:
            dt = clock.tick(60)
            move_timer += dt
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                new_direction = self.input_handler.handle_event(event)
                if new_direction:
                    direction = new_direction
            if move_timer >= move_delay:
                self.snake.move(direction)
                move_timer = 0
            self.renderer.draw_board()
            self.food.draw_food(self.screen)
            self.snake.draw_snake(self.screen)
            CollisionRules.check_food_collision(self.food, self.snake)
            game_status = CollisionRules.check_game_over(self.snake, self.board)
            if game_status == GameStatus.GAME_OVER:
                self.status = GameStatus.GAME_OVER

            pg.display.flip()
        pg.quit()

class Snake:
    # Manages body segments, movement, growth, and self‑collision checks.
    def __init__(self):
        self.snake = [((SCREEN_WIDTH // CELL_SIZE // 2) * CELL_SIZE, (SCREEN_HEIGHT // CELL_SIZE // 2) * CELL_SIZE)]
        self.snake_head = self.snake[0]
        self.skin = pg.Surface((CELL_SIZE, CELL_SIZE))
        self.skin.fill((255, 255, 255))
    
    def draw_snake(self, screen):
        for segment in self.snake:
            screen.blit(self.skin, segment)

    def move(self, direction):
        match direction:
            case Direction.UP:
                new_head = (self.snake_head[0], self.snake_head[1] - CELL_SIZE)

            case Direction.DOWN:
                new_head = (self.snake_head[0], self.snake_head[1] + CELL_SIZE)

            case Direction.LEFT:
                new_head = (self.snake_head[0] - CELL_SIZE, self.snake_head[1])

            case Direction.RIGHT:
                new_head = (self.snake_head[0] + CELL_SIZE, self.snake_head[1])

            case _:
                return
        self.snake.insert(0, new_head)
        self.snake.pop()
        self.snake_head = new_head
    
    def grow(self):
        tail = self.snake[-1]
        self.snake.append(tail)

class Food:
    # Holds position and respawn logic (validated against Snake).
    def __init__(self, board, snake):
        self.board = board
        self.snake = snake
        self.color = FOOD_COLOR
        self.position = self.spawn_food()

    def spawn_food(self):
        while True:
            position_x = rd.randint(0, (self.board.get_width() // CELL_SIZE) - 1) * CELL_SIZE
            position_y = rd.randint(0, (self.board.get_height() // CELL_SIZE) - 1) * CELL_SIZE
            position = (position_x, position_y)
            if position not in self.snake.snake:
                self.position = position
                return self.position

    def draw_food(self, screen):
        food_rectangle = pg.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE)
        pg.draw.rect(screen, self.color, food_rectangle)
    
    def delete_food(self):
        self.position = self.spawn_food()

class Renderer:
    # Draws game state; keep rendering separate from logic.
    def __init__(self, screen, board):
        self.screen = screen
        self.board = board
    
    def draw_board(self):
        self.screen.fill((0, 0, 0))
        cell_size = self.board.get_cell_size()
        board_width = self.board.get_width()
        board_height = self.board.get_height()

        for x in range(0, board_width, cell_size):
            pg.draw.line(self.screen, (40, 40, 40), (x, 0), (x, board_height))
        
        for y in range(0, board_height, cell_size):
            pg.draw.line(self.screen, (40, 40, 40), (0, y), (board_width, y))

class InputHandler:
    # Maps keys (or later agent actions) to movement commands.
    def handle_event(self, event):
        if event.type == pg.KEYDOWN:
            match event.key:
                case pg.K_UP:
                    return Direction.UP
                case pg.K_DOWN:
                    return Direction.DOWN
                case pg.K_LEFT:
                    return Direction.LEFT
                case pg.K_RIGHT:
                    return Direction.RIGHT
        return None
class GameState:
    # Snapshot of current state for ML (snake, food, score, etc.).
    pass

class CollisionRules:
    def check_food_collision(food, snake):
        if snake.snake_head == food.position:
            snake.grow()
            food.delete_food()
            return EventType.ATE_FOOD
        return None
    
    def check_wall_collision(snake, board):
        TOP_WALL = snake.snake_head[1] < 0
        BOTTOM_WALL = snake.snake_head[1] >= board.get_height()
        LEFT_WALL = snake.snake_head[0] < 0
        RIGHT_WALL = snake.snake_head[0] >= board.get_width()

        if TOP_WALL or BOTTOM_WALL or LEFT_WALL or RIGHT_WALL:
            return EventType.HIT_WALL
        return None

    def check_self_collision(snake):
        # Only check for self-collision if snake is longer than 2 tiles
        if len(snake.snake) > 2:
            for segment in snake.snake[1:]:
                if snake.snake_head == segment:
                    return EventType.HIT_SELF
        return None
    
    def check_win():
        pass
    
    def check_game_over(snake, board):
        if CollisionRules.check_wall_collision(snake, board) or CollisionRules.check_self_collision(snake):
            return GameStatus.GAME_OVER

if __name__ == "__main__":
    game = Game()
    game.run()
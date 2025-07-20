import pygame #for game visualisation
import random #to randomly  place food
import numpy as np #for Q-table operation
import sys #for system exit

#initializes pygame, sets up a 400x400 window with 20 pixelblocks
pygame.init()

WIDTH, HEIGHT = 400,400
BLOCK_SIZE = 20
ROWS, COLS = HEIGHT // BLOCK_SIZE, WIDTH // BLOCK_SIZE

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake RL")
clock = pygame.time.Clock()

#now creating a snake environment class
#initializes the snake at (5,5), direction -> up
#_place_food ensures the food is placed away from the snake

class SnakeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake = [(5, 5)]
        self.direction = (0, -1)
        self.food = self._place_food()
        self.score = 0
        return self._get_state()
    
    def _place_food(self):
        while True:
            food = (random.randint(0, COLS-1), random.randint(0, ROWS-1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        head = self.snake[0]
        
        # Create a more informative state representation
        # Direction to food
        food_dir_x = np.sign(self.food[0] - head[0])
        food_dir_y = np.sign(self.food[1] - head[1])
        
        # Danger detection in all 4 directions
        danger = [0, 0, 0, 0]  # UP, RIGHT, DOWN, LEFT
        
        # Check each direction for danger
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = head[0] + dx, head[1] + dy
            
            # Check wall collision
            if not (0 <= new_x < COLS and 0 <= new_y < ROWS):
                danger[i] = 1
            # Check self collision
            elif (new_x, new_y) in self.snake:
                danger[i] = 1
        
        # Current direction (one-hot encoded)
        current_dir = [0, 0, 0, 0]
        if self.direction == (0, -1): current_dir[0] = 1  # UP
        elif self.direction == (1, 0): current_dir[1] = 1  # RIGHT
        elif self.direction == (0, 1): current_dir[2] = 1  # DOWN
        elif self.direction == (-1, 0): current_dir[3] = 1  # LEFT
        
        # Combine all state information
        state = (food_dir_x, food_dir_y) + tuple(danger) + tuple(current_dir)
        return state
    
    def step(self, action):
        self._change_direction(action)
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        reward = -0.1
        done = False

        # Check wall collision
        if not (0 <= new_head[0] < COLS and 0 <= new_head[1] < ROWS):
            reward = -10
            done = True
        # Check self collision
        elif new_head in self.snake:
            reward = -10
            done = True
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 10
                self.food = self._place_food()
                self.score += 1
            else:
                self.snake.pop()

        return self._get_state(), reward, done
    
    def _change_direction(self, action):
        if action == 0: self.direction = (0, -1) #UP
        elif action == 1: self.direction = (1, 0) #RIGHT
        elif action == 2: self.direction = (0, 1) #DOWN
        elif action == 3: self.direction = (-1, 0) #LEFT
    
    def render(self):
        screen.fill(BLACK)
        for x, y in self.snake:
            pygame.draw.rect(screen, GREEN, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(screen, RED, (self.food[0] * BLOCK_SIZE, self.food[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        # Display score in white text
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

#Q table initialization
actions = [0, 1, 2, 3] #UP, RIGHT, DOWN, LEFT
q_table = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.1

#Q-table helper
# initializes unseen states with zero Q-values.
def get_q(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(actions))
    return q_table[state]

#Choose Action e-greedy
#With probability epsilon, choose random action(exploration)
def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)
    return np.argmax(get_q(state))

#CORE Training loop
#updates Q-values using Q-learning formula

game = SnakeGame()
episodes = 1000

print("Starting training...")
for episode in range(episodes):
    state = game.reset()
    total_reward = 0
    done = False
    steps = 0
    max_steps = 1000  # Prevent infinite loops

    while not done and steps < max_steps:
        action = choose_action(state)
        next_state, reward, done = game.step(action)

        q_old = get_q(state)[action]
        q_next = max(get_q(next_state))
        q_table[state][action] = q_old + alpha * (reward + gamma * q_next - q_old)

        state = next_state
        total_reward += reward
        steps += 1
    
    if episode % 100 == 0:
        print(f"Episode {episode+1}: Score = {game.score}, Total Reward = {round(total_reward, 2)}")

print("Training completed! Starting visualization...")

#Visualize Final Policy
epsilon = 0  # No exploration during visualization
while True:
    state = game.reset()
    done = False
    steps = 0
    max_steps = 1000

    while not done and steps < max_steps:
        game.render()
        action = choose_action(state)
        state, _, done = game.step(action)
        clock.tick(10)  # 10 FPS
        steps += 1
    
    pygame.time.wait(1000)  # Wait 1 second between games


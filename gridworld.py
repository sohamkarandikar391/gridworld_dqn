import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class GridWorld:
    def __init__(self, rows, cols, walls, terminals, start_pos, step_reward, action_prob):
        self.rows = rows
        self.cols = cols    
        self.walls = walls
        self.terminals = terminals
        self.start_pos = start_pos
        self.step_reward = step_reward
        self.good = True
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]

        for wall in walls:
            row, col = wall
            self.grid[row][col] = -1

        self.terminal_states = set()
        self.terminal_rewards = {}
        for (row,col), reward in terminals.items():
            self.grid[row][col] = 1
            self.terminal_states.add((row, col))
            self.terminal_rewards[(row, col)] = reward

        self.state_to_pos = {}
        self.pos_to_state = {}

        state_num = 0

        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] != -1:
                    self.state_to_pos[state_num] = (row, col)
                    self.pos_to_state[(row, col)] = state_num
                    state_num += 1

        self.num_states = state_num
        self.states = range(self.num_states)
        self.actions = {0, 1, 2, 3}
        self.action_effects = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        self.message = ""
        self.transitions = {}
        self.action_prob = action_prob

        for state in range(self.num_states):
            if state not in self.transitions:
                self.transitions[state] = {}

            current_pos = self.state_to_pos[state]

            if current_pos in self.terminal_states:
                continue

            for action in self.actions:
                row_change, col_change = self.action_effects[action]
                new_row = current_pos[0] + row_change
                new_col = current_pos[1] + col_change

                if(new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols or self.grid[new_row][new_col] == -1):
                    next_state = state
                else:
                    next_state = self.pos_to_state[(new_row, new_col)]

                self.transitions[state][action] = (next_state, self.step_reward)
        
        self.current_state = self.pos_to_state[self.start_pos]
        self.agent_pos = self.start_pos
        self.done = False
        
        if (self.start_pos[0] < 0 or self.start_pos[0] >= self.rows or 
            self.start_pos[1] < 0 or self.start_pos[1] >= self.cols or
            self.grid[self.start_pos[0]][self.start_pos[1]] == -1):
            raise ValueError(f"Invalid start position {self.start_pos}: out of bounds or wall")

    def step(self, action):
        if self.done:
            return self.current_state, 0, True, {}
        
        # self.probs = list(self.action_prob[1:])
        self.probs = [0] * len(self.actions)
        k = action
        for i in range(4):
            self.probs[i] = self.probs[i] + self.action_prob[-k + i]        

        # print(self.probs)
        self.actual_action = random.choices(list(self.actions), weights=self.probs)[0]
        next_state, reward = self.transitions[self.current_state][self.actual_action]
        
        if next_state == self.current_state:
            reward = 0
        next_pos = self.state_to_pos[next_state]
        if next_pos in self.terminal_states:

            reward = self.terminal_rewards[next_pos]
            if self.terminal_rewards[next_pos] < 0:
                # print("Bad reward" , self.terminal_rewards[next_pos])
                self.good = False
            self.done = True
            # print(self.good)
        self.current_state = next_state
        self.agent_pos = next_pos
        return self.current_state, reward, self.done, {}
    
    def reset(self):
        self.agent_pos = self.start_pos
        self.current_state = self.pos_to_state[self.agent_pos]
        self.done = False
        self.good = True
        return self.current_state
    
    def render(self, interactive=False):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.ax.set_xlim(-0.5, self.cols - 0.5)
            self.ax.set_ylim(-0.5, self.rows - 0.5)
            self.ax.set_aspect('equal')
            
            if interactive:
                self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.cols - 0.5)
        self.ax.set_ylim(-0.5, self.rows - 0.5)
        self.ax.set_aspect('equal')
        
        for row in range(self.rows):
            for col in range(self.cols):
                display_row = row
                
                if self.grid[row][col] == -1:
                    self.ax.add_patch(patches.Rectangle((col-0.5, display_row-0.5), 1, 1, facecolor='black'))
                elif (row, col) in self.terminal_states:
                    if self.terminal_rewards[(row, col)] > 0:
                        self.ax.add_patch(patches.Rectangle((col-0.5, display_row-0.5), 1, 1, facecolor='green'))
                    else:
                        self.ax.add_patch(patches.Rectangle((col-0.5, display_row-0.5), 1, 1, facecolor='red'))
                else:
                    self.ax.add_patch(patches.Rectangle((col-0.5, display_row-0.5), 1, 1, facecolor='white', edgecolor='gray'))
        
        agent_row, agent_col = self.agent_pos

        self.ax.plot(agent_col, agent_row, 'bo', markersize=15)

        self.ax.set_title('GridWorld Environment', fontsize=16, fontweight='bold', pad=20)


        if self.message:
            self.ax.text(0, -0.3, self.message, fontsize=12, color='black')
        
        if self.done:
            if self.good == False:
                self.ax.text(0, -0.1, "You've fallen down the hill. Press R to reset.", fontsize = 10, color='red')
                self.good = True
            else: 
                self.ax.text(0, -0.1, "Congrats, you've reached the goal. Press R to reset.", fontsize=10, color='green')
                self.good = True
        
        self.ax.set_xticks(range(self.cols))
        self.ax.set_yticks(range(self.rows))
        self.ax.grid(True, alpha=0.3)
        
        if interactive:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.draw()
            plt.pause(0.01)

    
    def _on_key_press(self, event):
        if self.done and event.key == 'r':
            self.reset()
            self.render(interactive=True)
        elif not self.done:
            action = None
            if event.key == 'up':
                action = 0
            elif event.key == 'down':
                action = 2
            elif event.key == 'right':
                action = 1
            elif event.key == 'left':
                action = 3
                
            if action is not None:
                state, reward, done, _ = self.step(action)
                # print(state, reward, done)
                self.render(interactive=True)

    def get_states(self):
        return self.states
    
    def get_actions(self):
        return self.actions
    
    def get_current_state(self):
        return self.current_state
    
    def get_current_pos(self):
        return self.state_to_pos[self.current_state]
    
    def close(self):
       if hasattr(self, 'fig'):
           plt.close(self.fig)
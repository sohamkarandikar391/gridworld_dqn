# main.py
from gridworld import GridWorld
from dqnagent import DQNAgent
from train import train_dqn, test_agent, plot_training_stats
import matplotlib.pyplot as plt

# 1. Create the GridWorld environment
walls = [(1, 1)]
terminals = {(2, 3): 1, (1, 3): -1}
action_probs = [0.8, 0.1, 0, 0.1] # 80% intended and 10$ Left and 10% right
env = GridWorld(3, 4, walls, terminals, (0, 0), -0.04, action_probs)


print(f"Environment created with {env.num_states} states")

# 2. Create the DQN Agent
agent = DQNAgent(
    state_size=env.num_states,
    action_size=4,
    hidden_layers=[64, 64],  # Two hidden layers with 64 neurons each
    learning_rate=0.001,
    gamma=0.95,
    buffer_capacity=5000,
    batch_size=64,
    target_update_freq=20,
    min_buffer_size=1000
)

print("DQN Agent created")
hidden_str = " -> ".join(map(str, agent.hidden_layers))
print(f"Q-Network architecture: {env.num_states} -> {hidden_str} -> {agent.action_size}")

# 3. Train the agent
print("\nStarting training...\n")
stats = train_dqn(
    env=env,
    agent=agent,
    num_episodes=2000,
    max_steps_per_episode=100,
    print_every=50,
    test_every=200,
    num_test_episodes=20
)

print("\nTraining completed!")

# 4. Plot the results
plot_training_stats(stats, save_path='dqn_training_results.png')

# 5. Final test
print("\nRunning final evaluation...")
final_test_rewards, final_test_lengths, final_success_rate = test_agent(
    env, agent, num_test_episodes=100, max_steps_per_episode=80
)

print(f"\nFinal Test Results (100 episodes):")
print(f"  Average Reward: {sum(final_test_rewards)/len(final_test_rewards):.2f}")
print(f"  Average Length: {sum(final_test_lengths)/len(final_test_lengths):.2f}")
print(f"  Success Rate: {final_success_rate:.1f}%")

# 6. Visualize the agent playing
print("\nWatch the trained agent play...")
agent.epsilon = 0.0  # No exploration, pure exploitation

num_demo_episodes = 2
for episode in range(num_demo_episodes):
    print(f"\nDemo Episode {episode + 1}/{num_demo_episodes}")
    state = env.reset()
    env.render(interactive=False)
    
    total_reward = 0
    steps = 0
    
    for step in range(80):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        env.render(interactive=False)
        
        state = next_state
        total_reward += reward
        steps += 1
        plt.pause(0.5)
        if done:
            print(f"Episode finished in {steps} steps with reward {total_reward:.2f}")
            import time
            time.sleep(2)  # Pause before next episode
            break

env.close()

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import random
from gridworld import GridWorld
from dqnagent import DQNAgent
from train import train_dqn, test_agent # Make sure train.py is accessible

# --- Experiment Configuration ---
NUM_RUNS = 5
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 100
TEST_EVERY = 20
NUM_TEST_EPISODES = 10

# --- Helper Functions to create/re-initialize models ---

def set_seeds(seed_value):
    """Sets the seed for all relevant random number generators."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

def create_environment():
    """Creates a fresh instance of the GridWorld environment."""
    walls = [(1, 1)]
    terminals = {(2, 3): 1, (1, 3): -1}
    action_probs = [0.8, 0.1, 0, 0.1]
    return GridWorld(3, 4, walls, terminals, (0, 0), -0.04, action_probs)

def create_agent(env):
    """Creates a fresh instance of the DQNAgent."""
    agent = DQNAgent(
        state_size=env.num_states,
        action_size=4,
        hidden_layers=[64, 64],   
        learning_rate=0.001,
        gamma=0.95,
        buffer_capacity=5000,
        batch_size=64,
        target_update_freq=50,     
        min_buffer_size=1000      
    )
    return agent

# --- Aggregate and Plotting Function ---

def plot_aggregate_stats(train_data, test_data, test_x_indices, title, y_label, save_path):
    """
    Aggregates data from all runs and plots the mean and std deviation.
    """
    # --- Aggregate Training Data ---
    train_np = np.array(train_data)
    mean_train = np.mean(train_np, axis=0)
    std_train = np.std(train_np, axis=0)
    train_x = np.arange(1, len(mean_train) + 1) # X-axis for training

    # --- Aggregate Testing Data ---
    test_np = np.array(test_data)
    mean_test = np.mean(test_np, axis=0)
    std_test = np.std(test_np, axis=0)

    # --- Create Plot ---
    plt.figure(figsize=(12, 7))

    # Plot Training (Blue)
    plt.plot(train_x, mean_train, 'b-', linewidth=2.5, label='Training (Avg)')
    plt.fill_between(train_x, mean_train - std_train, mean_train + std_train,
                     color='blue', alpha=0.2, label='Training (Std Dev)')

    # Plot Testing (Red) - No marker
    plt.plot(test_x_indices, mean_test, 'r-', linewidth=2.5, label='Testing (Avg)')
    
    plt.fill_between(test_x_indices, mean_test - std_test, mean_test + std_test,
                     color='red', alpha=0.2, label='Testing (Std Dev)')

    plt.title(f'{title} (Avg. over {NUM_RUNS} Runs)', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


# --- Main Experiment Loop ---

print(f"Starting {NUM_RUNS} training runs...")

# Lists to store the stats from all runs
all_train_rewards = []
all_train_steps = []
all_test_rewards = []
all_test_steps = []

test_episode_indices = None
last_trained_env = None
last_trained_agent = None

for run in range(NUM_RUNS):
    # Set a different seed for each run
    seed_value = run # Use 0, 1, 2, 3, 4 as seeds
    set_seeds(seed_value)
    print(f"\n--- Starting Run {run + 1}/{NUM_RUNS} (Seed: {seed_value}) ---")
    
    env = create_environment()
    agent = create_agent(env)

    stats = train_dqn(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        print_every=500,
        test_every=TEST_EVERY,
        num_test_episodes=NUM_TEST_EPISODES
    )

    all_train_rewards.append(stats['episode_rewards'])
    all_train_steps.append(stats['episode_lengths'])
    all_test_rewards.append(stats['test_rewards'])
    all_test_steps.append(stats['test_lengths'])

    if test_episode_indices is None:
        test_episode_indices = stats['test_episodes']

    if run == NUM_RUNS - 1:
        last_trained_env = env
        last_trained_agent = agent

print("\nAll training runs complete. Aggregating results...")

# --- Generate the two plots ---

# Plot 1: Rewards
plot_aggregate_stats(
    all_train_rewards,
    all_test_rewards,
    test_episode_indices,
    'Average Reward per Episode',
    'Total Reward',
    'dqn_rewards_plot.png'
)

# Plot 2: Steps
plot_aggregate_stats(
    all_train_steps,
    all_test_steps,
    test_episode_indices,
    'Average Steps per Episode',
    'Steps per Episode',
    'dqn_steps_plot.png'
)

# --- Final Evaluation (from your original main.py) ---
print("\nRunning final evaluation on the *last* trained agent...")
final_test_rewards, final_test_lengths, final_success_rate = test_agent(
    last_trained_env, last_trained_agent,
    num_test_episodes=100, max_steps_per_episode=80
)

print(f"\nFinal Test Results (100 episodes):")
print(f"  Average Reward: {np.mean(final_test_rewards):.2f}")
print(f"  Average Length: {np.mean(final_test_lengths):.2f}")
print(f"  Success Rate: {final_success_rate:.1f}%")

# --- Visualize the last agent playing ---
print("\nWatch the *last* trained agent play...")
if last_trained_agent:
    last_trained_agent.epsilon = 0.0  # No exploration
    for episode in range(2):
        print(f"\nDemo Episode {episode + 1}/{2}")
        state = last_trained_env.reset()
        last_trained_env.render(interactive=False)

        total_reward = 0
        steps = 0
        for step in range(80):
            action = last_trained_agent.select_action(state)
            next_state, reward, done, _ = last_trained_env.step(action)
            last_trained_env.render(interactive=False)
            state = next_state
            total_reward += reward
            steps += 1
            plt.pause(0.5)
            if done:
                print(f"Episode finished in {steps} steps with reward {total_reward:.2f}")
                time.sleep(2)
                break

if last_trained_env:
    last_trained_env.close()
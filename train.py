from gridworld import GridWorld
from dqnagent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np


def test_agent(env, agent, num_test_episodes=10, max_steps_per_episode=200):
        """
        Test the trained agent without exploration (epsilon=0)
        
        Parameters:
        - env: GridWorld environment
        - agent: DQNAgent instance
        - num_test_episodes: number of episodes to test
        - max_steps_per_episode: maximum steps per episode
        
        Returns:
        - test_rewards: list of rewards for each test episode
        - test_lengths: list of episode lengths
        - success_rate: percentage of episodes that reached positive terminal state
        """
        test_rewards = []
        test_lengths = []
        successes = 0
        
        # Save current epsilon and set to 0 (no exploration)
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        for episode in range(num_test_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Select action (greedy, no exploration)
                action = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if done:
                    if reward > 0:  # Reached good terminal state
                        successes += 1
                    break
            
            test_rewards.append(episode_reward)
            test_lengths.append(steps)
        
        # Restore original epsilon
        agent.epsilon = original_epsilon
        
        success_rate = (successes / num_test_episodes) * 100
        
        return test_rewards, test_lengths, success_rate

def train_dqn(env, agent, num_episodes=1000, max_steps_per_episode=80, print_every=100, test_every=100, num_test_episodes=10):
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    epsilons = []

    test_episodes = []
    test_rewards_history = []
    test_lengths_history = []
    test_success_rates = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        steps = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)

            loss = agent.train()

            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break
        
        agent.decay_epsilon()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        epsilons.append(agent.epsilon)

        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        if (episode + 1) % test_every == 0:
            test_rewards, test_lengths, success_rate = test_agent(
                env, agent, num_test_episodes, max_steps_per_episode
            )
            
            # Store test statistics
            test_episodes.append(episode + 1)
            test_rewards_history.append(np.mean(test_rewards))
            test_lengths_history.append(np.mean(test_lengths))
            test_success_rates.append(success_rate)
            
            print(f"Episode {episode + 1}/{num_episodes} - TEST RESULTS")
            print(f"  Test Avg Reward: {np.mean(test_rewards):.2f}")
            print(f"  Test Avg Length: {np.mean(test_lengths):.2f}")
            print(f"  Test Success Rate: {success_rate:.1f}%")
            print(f"  Training Epsilon: {agent.epsilon:.3f}")
            print()
        
        # Print training progress
        elif (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Train Avg Reward: {avg_reward:.2f}")
            print(f"  Train Avg Length: {avg_length:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print()

    stats = {
    'episode_rewards': episode_rewards,
    'episode_lengths': episode_lengths,
    'episode_losses': episode_losses,
    'epsilons': epsilons,
    'test_episodes': test_episodes,
    'test_rewards': test_rewards_history,
    'test_lengths': test_lengths_history,
    'test_success_rates': test_success_rates
    }
    
    return stats


import matplotlib.pyplot as plt
import numpy as np

def plot_training_stats(stats, save_path=None, window=50):
    """
    Plot training and testing statistics with moving averages
    
    Parameters:
    - stats: dictionary returned by train_dqn()
    - save_path: optional path to save the figure
    - window: window size for moving average
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('DQN Training and Testing Statistics', fontsize=16, fontweight='bold')
    
    # 1. Rewards: Training and Testing together
    ax1 = axes[0]
    
    episodes_train = np.arange(len(stats['episode_rewards']))
    ax1.plot(episodes_train, stats['episode_rewards'], alpha=0.3, color='blue', label='Training Reward')
    
    if len(stats['episode_rewards']) >= window:
        moving_avg_train = np.convolve(stats['episode_rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(stats['episode_rewards'])), moving_avg_train, 
                color='blue', linewidth=2, alpha=0.8, label=f'Training {window}-Ep Avg')
    
    ax1.scatter(stats['test_episodes'], stats['test_rewards'], alpha=0.3, color='green', s=50, label='Test Reward')
    
    if len(stats['test_rewards']) >= 3:
        test_window = min(3, len(stats['test_rewards']))
        test_moving_avg = np.convolve(stats['test_rewards'], np.ones(test_window)/test_window, mode='valid')
        test_episodes_avg = stats['test_episodes'][test_window-1:]
        ax1.plot(test_episodes_avg, test_moving_avg, color='green', linewidth=2, alpha=0.8,
                marker='o', markersize=6, label=f'Test {test_window}-Point Avg')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training vs Testing Rewards', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Lengths: Training and Testing together
    ax2 = axes[1]
    
    ax2.plot(episodes_train, stats['episode_lengths'], alpha=0.3, color='red', label='Training Steps')
    
    if len(stats['episode_lengths']) >= window:
        moving_avg_train_len = np.convolve(stats['episode_lengths'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(stats['episode_lengths'])), moving_avg_train_len, 
                color='red', linewidth=2, alpha=0.8, label=f'Training {window}-Ep Avg')
    
    ax2.scatter(stats['test_episodes'], stats['test_lengths'], alpha=0.3, color='orange', s=50, label='Test Steps')
    
    if len(stats['test_lengths']) >= 3:
        test_window = min(3, len(stats['test_lengths']))
        test_moving_avg_len = np.convolve(stats['test_lengths'], np.ones(test_window)/test_window, mode='valid')
        test_episodes_avg_len = stats['test_episodes'][test_window-1:]
        ax2.plot(test_episodes_avg_len, test_moving_avg_len, color='orange', linewidth=2, alpha=0.8,
                marker='o', markersize=6, label=f'Test {test_window}-Point Avg')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps', fontsize=12)
    ax2.set_title('Training vs Testing Episode Lengths', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Loss
    ax3 = axes[2]
    
    ax3.plot(episodes_train, stats['episode_losses'], alpha=0.3, color='purple', label='Loss')
    
    if len(stats['episode_losses']) >= window:
        moving_avg_loss = np.convolve(stats['episode_losses'], np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(stats['episode_losses'])), moving_avg_loss, 
                color='purple', linewidth=2, alpha=0.8, label=f'{window}-Episode Avg')
    
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Training Loss', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
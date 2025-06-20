# ai_book



Playing Atari with Deep Reinforcement Learning - https://arxiv.org/pdf/1312.5602

Prioritized Experience Replay - https://arxiv.org/abs/1511.05952

# Double DQN Hyperparameters: Common Values & Relationships

## Core Hyperparameters

### Learning Rate (α)
**Common Values:** 0.00025 - 0.001
**Default:** 0.0005
**Why:**
- Too high (>0.01): Unstable learning, oscillations
- Too low (<0.0001): Very slow convergence
- 0.0005 provides good balance for most environments

### Discount Factor (γ)
**Common Values:** 0.95 - 0.99
**Default:** 0.99
**Why:**
- Higher values (0.99): Long-term planning, suitable for most RL tasks
- Lower values (0.95): Short-term focus, good for episodic tasks
- 0.99 works well for most continuous/long-horizon tasks

### Epsilon (Exploration)
**Initial:** 1.0
**Final:** 0.01 - 0.1
**Decay:** 0.995 - 0.9999 per step
**Why:**
- Start with full exploration (ε=1.0)
- End with small exploration (ε=0.01-0.1) to maintain some randomness
- Gradual decay balances exploration vs exploitation

### Batch Size
**Common Values:** 32, 64, 128
**Default:** 32
**Why:**
- 32: Good for most environments, stable gradients
- 64-128: Better for complex environments, more stable but slower
- Power of 2 for computational efficiency

### Replay Buffer Size
**Common Values:** 10,000 - 1,000,000
**Default:** 100,000
**Why:**
- Small buffers (<10K): Fast but limited diversity
- Large buffers (>100K): More diverse but memory intensive
- 100K provides good balance for most tasks

### Target Network Update Frequency
**Common Values:** 1,000 - 10,000 steps
**Default:** 4,000 steps
**Why:**
- Too frequent (<1000): Unstable targets
- Too infrequent (>10K): Slow adaptation
- 4,000 steps balances stability and learning speed

## Network Architecture

### Hidden Layers
**Common:** 2-3 fully connected layers
**Sizes:** [128, 128] or [256, 256] or [512, 256, 128]
**Why:**
- 2-3 layers sufficient for most RL tasks
- 128-512 neurons provide good representational capacity
- Avoid too deep networks (harder to train in RL)

### Activation Functions
**Common:** ReLU, Leaky ReLU
**Default:** ReLU
**Why:**
- ReLU: Simple, effective, no vanishing gradients
- Leaky ReLU: Handles dead neurons better
- Avoid sigmoid/tanh (vanishing gradients)

## Environment-Specific Defaults

### Atari Games
```
Learning Rate: 0.00025
Discount Factor: 0.99
Epsilon: 1.0 → 0.1 (decay over 1M steps)
Batch Size: 32
Buffer Size: 1,000,000
Target Update: 4,000 steps
Network: [512, 512] or CNN
```

### Control Tasks (CartPole, MountainCar)
```
Learning Rate: 0.001
Discount Factor: 0.95
Epsilon: 1.0 → 0.01 (decay over 100K steps)
Batch Size: 64
Buffer Size: 50,000
Target Update: 2,000 steps
Network: [128, 128]
```

### Continuous Control / Robotics
```
Learning Rate: 0.0003
Discount Factor: 0.99
Epsilon: 1.0 → 0.05 (decay over 500K steps)
Batch Size: 128
Buffer Size: 200,000
Target Update: 8,000 steps
Network: [256, 256, 128]
```

## Critical Hyperparameter Relationships

### 1. Learning Rate ↔ Target Update Frequency
```
High Learning Rate (0.001+) → More Frequent Updates (1,000-2,000 steps)
- Fast changes require frequent target updates for stability

Low Learning Rate (0.0001) → Less Frequent Updates (8,000-20,000 steps)  
- Slow changes can tolerate stale targets longer
```

### 2. Learning Rate ↔ Batch Size
```
High Learning Rate → Larger Batch Size (64-128)
- Larger batches provide more stable gradients for aggressive learning

Low Learning Rate → Smaller Batch Size (16-32)
- Smaller batches acceptable when learning cautiously
```

### 3. Buffer Size ↔ Batch Size
```
Large Buffer (>100K) → Can use Larger Batch (64-128)
- More diverse sampling possible

Small Buffer (<50K) → Should use Smaller Batch (16-32)
- Limited diversity, avoid overfitting to recent experiences
```

### 4. Epsilon Decay ↔ Environment Complexity
```
Simple Environments → Faster Decay (0.995/step)
- Quick convergence to exploitation

Complex Environments → Slower Decay (0.9995/step)
- Need longer exploration phase
```

### 5. Discount Factor ↔ Episode Length
```
Short Episodes (<100 steps) → Lower Gamma (0.95-0.97)
- Less emphasis on distant future

Long Episodes (>1000 steps) → Higher Gamma (0.99-0.995)
- Important to consider long-term consequences
```

### 6. Network Size ↔ Environment Complexity
```
Simple State Spaces → Smaller Networks [64, 64]
- CartPole, simple control tasks

Complex State Spaces → Larger Networks [256, 256] or CNN
- Atari, robotics, high-dimensional inputs
```

## Advanced Relationships

### Training Stability Triangle
```
Learning Rate ↑ → Target Update Freq ↑ → Batch Size ↑
```
These three must scale together for stable training.

### Sample Efficiency Trade-offs
```
Buffer Size ↑ + Batch Size ↑ = Better Sample Efficiency
BUT
Memory Usage ↑ + Training Time ↑
```

### Exploration-Exploitation Balance
```
Final Epsilon ↔ Learning Rate ↔ Target Update Frequency
```
Aggressive learning needs more exploration maintenance.

## Hyperparameter Tuning Order

### 1. Start with Environment Defaults
Use the environment-specific defaults above as starting point.

### 2. Tune Core Parameters First
```
1. Learning Rate (most critical)
2. Target Update Frequency  
3. Batch Size
4. Epsilon Decay Schedule
```

### 3. Fine-tune Secondary Parameters
```
5. Buffer Size
6. Network Architecture
7. Final Epsilon Value
8. Discount Factor (usually keep at 0.99)
```

### 4. Advanced Optimizations
```
9. Gradient clipping
10. Learning rate scheduling
11. Prioritized experience replay parameters
```

## Common Pitfalls & Solutions

### Problem: Training Instability
**Symptoms:** Reward oscillations, loss spikes
**Solutions:**
- Decrease learning rate (0.001 → 0.0005)
- Increase target update frequency (2000 → 4000 steps)
- Increase batch size (32 → 64)

### Problem: Slow Learning
**Symptoms:** Flat reward curves, slow improvement
**Solutions:**
- Increase learning rate (0.0005 → 0.001)
- Decrease target update frequency (4000 → 2000 steps)
- Check epsilon decay (ensure sufficient exploration)

### Problem: Poor Final Performance
**Symptoms:** Good learning but suboptimal final policy
**Solutions:**
- Decrease final epsilon (0.1 → 0.01)
- Increase buffer size for more diverse experiences
- Tune discount factor for environment horizon

## Quick Start Recommendations

### For Most Environments:
```python
hyperparameters = {
    'learning_rate': 0.0005,
    'discount_factor': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'batch_size': 32,
    'buffer_size': 100_000,
    'target_update_freq': 4_000,
    'network_layers': [128, 128]
}
```

### Tuning Strategy:
1. Start with defaults
2. Monitor training stability
3. Adjust learning rate first if needed
4. Scale other parameters proportionally
5. Fine-tune based on specific environment feedback

Remember: **Hyperparameter relationships are more important than individual values**. Always consider how changes to one parameter should influence others!


# Double DQN Monitoring: Key Metrics & Analysis

## Core Performance Metrics

### 1. Episode Reward
**What it measures:** Total reward accumulated per episode
**Why critical:** Direct measure of agent performance
**How to monitor:**
```python
# Log every episode
episode_rewards = []
episode_rewards.append(total_episode_reward)

# Track moving averages
reward_ma_100 = np.mean(episode_rewards[-100:])  # Last 100 episodes
reward_ma_1000 = np.mean(episode_rewards[-1000:])  # Last 1000 episodes
```

**What to look for:**
- **Upward trend**: Good learning progress
- **Plateauing**: May need hyperparameter adjustment
- **Oscillations**: Potential instability (check learning rate)
- **Sudden drops**: Possible catastrophic forgetting

### 2. Episode Length
**What it measures:** Number of steps per episode
**Why important:** Indicates learning efficiency and task mastery
**How to monitor:**
```python
episode_lengths = []
episode_lengths.append(episode_step_count)
avg_length = np.mean(episode_lengths[-100:])
```

**What to look for:**
- **Increasing length**: Agent surviving longer (good for survival tasks)
- **Decreasing length**: Agent solving faster (good for goal-reaching tasks)
- **High variance**: Inconsistent policy (may need more training)

## Training Stability Metrics

### 3. Q-Loss (MSE Loss)
**What it measures:** How well Q-values match targets
**Why critical:** Core learning signal, indicates training stability
**How to monitor:**
```python
# During training step
current_q = q_network(states).gather(1, actions)
target_q = rewards + gamma * max(target_network(next_states))
loss = F.mse_loss(current_q, target_q)

# Log every training step or batch
losses.append(loss.item())
loss_ma = np.mean(losses[-1000:])  # Moving average
```

**What to look for:**
- **Decreasing trend**: Good learning
- **Stable low values**: Converged learning
- **Sudden spikes**: Instability (check learning rate, target update freq)
- **Increasing trend**: Divergence (reduce learning rate)

### 4. TD Error Distribution
**What it measures:** Magnitude of prediction errors
**Why important:** Indicates prediction quality and training focus
**How to monitor:**
```python
td_errors = abs(current_q - target_q).detach().cpu().numpy()

# Track statistics
td_error_mean = np.mean(td_errors)
td_error_std = np.std(td_errors)
td_error_max = np.max(td_errors)

# Track distribution percentiles
td_error_p50 = np.percentile(td_errors, 50)
td_error_p95 = np.percentile(td_errors, 95)
```

**What to look for:**
- **Decreasing mean**: Improving predictions
- **High max values**: Outlier experiences (good for prioritized replay)
- **Narrow distribution**: Well-calibrated Q-values

### 5. Gradient Norms
**What it measures:** Magnitude of network updates
**Why important:** Indicates training stability and learning rate appropriateness
**How to monitor:**
```python
# After loss.backward() but before optimizer.step()
total_norm = 0
for p in q_network.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)

gradient_norms.append(total_norm)
```

**What to look for:**
- **Stable moderate values** (0.1-10): Good learning rate
- **Very small values** (<0.01): Learning rate too small
- **Very large values** (>100): Learning rate too large, risk of instability
- **Sudden spikes**: Potential gradient explosion

## Exploration Metrics

### 6. Epsilon Value
**What it measures:** Current exploration rate
**Why important:** Tracks exploration-exploitation balance
**How to monitor:**
```python
# Track epsilon decay
epsilons = []
epsilons.append(current_epsilon)

# Monitor exploration actions
exploration_rate = num_random_actions / total_actions
```

**What to look for:**
- **Smooth decay**: Proper exploration schedule
- **Too fast decay**: May miss important exploration
- **Too slow decay**: Inefficient exploitation phase

### 7. Action Distribution
**What it measures:** Frequency of each action taken
**Why important:** Indicates policy diversity and potential action bias
**How to monitor:**
```python
action_counts = defaultdict(int)
action_counts[action] += 1

# Periodically compute distribution
total_actions = sum(action_counts.values())
action_probs = {a: count/total_actions for a, count in action_counts.items()}
action_entropy = -sum(p * np.log(p) for p in action_probs.values() if p > 0)
```

**What to look for:**
- **Balanced early**: Good exploration
- **Concentrated later**: Policy convergence
- **Stuck on one action**: Potential exploration failure

## Q-Value Analysis Metrics

### 8. Q-Value Statistics
**What it measures:** Distribution and magnitude of Q-value estimates
**Why important:** Indicates value function quality and potential overestimation
**How to monitor:**
```python
with torch.no_grad():
    q_values = q_network(sample_states)
    
    q_mean = q_values.mean().item()
    q_std = q_values.std().item()
    q_max = q_values.max().item()
    q_min = q_values.min().item()
    
    # Track over time
    q_stats = {
        'mean': q_mean,
        'std': q_std, 
        'max': q_max,
        'min': q_min
    }
```

**What to look for:**
- **Reasonable magnitude**: Should relate to reward scale
- **Overestimation**: Q-values much larger than expected returns
- **Underestimation**: Q-values much smaller than actual returns
- **Growing without bound**: Potential instability

### 9. Target vs Current Q-Value Gap
**What it measures:** Difference between main and target network predictions
**Why important:** Indicates target network staleness and update frequency effectiveness
**How to monitor:**
```python
with torch.no_grad():
    current_q = q_network(states)
    target_q = target_network(states)
    
    q_gap = torch.mean(torch.abs(current_q - target_q)).item()
    q_gaps.append(q_gap)
```

**What to look for:**
- **Moderate gap**: Healthy difference between networks
- **Very small gap**: Target updates too frequent
- **Very large gap**: Target updates too infrequent

## Memory and Sampling Metrics

### 10. Replay Buffer Statistics
**What it measures:** Quality and diversity of stored experiences
**Why important:** Affects sample efficiency and learning stability
**How to monitor:**
```python
# Buffer utilization
buffer_utilization = len(replay_buffer) / replay_buffer.capacity

# Reward distribution in buffer
buffer_rewards = [transition.reward for transition in replay_buffer.buffer]
reward_diversity = np.std(buffer_rewards)

# State diversity (for simple state spaces)
if state_space_is_discrete:
    unique_states = len(set(transition.state for transition in replay_buffer.buffer))
    state_coverage = unique_states / total_possible_states
```

**What to look for:**
- **High utilization**: Buffer being used effectively
- **Reward diversity**: Good mix of experiences
- **State coverage**: Exploring state space well

## Environment-Specific Metrics

### 11. Success Rate (for Goal-Based Tasks)
**What it measures:** Percentage of episodes that reach the goal
**How to monitor:**
```python
successes = []
successes.append(1 if episode_reward > success_threshold else 0)
success_rate = np.mean(successes[-100:])  # Last 100 episodes
```

### 12. Convergence Metrics
**What it measures:** Whether learning has plateaued
**How to monitor:**
```python
# Reward stability
recent_rewards = episode_rewards[-100:]
reward_variance = np.var(recent_rewards)

# Policy stability (action consistency)
policy_change_rate = action_changes / total_actions
```

## Monitoring Implementation Strategy

### Real-Time Dashboards
```python
# Use tools like Weights & Biases, TensorBoard, or matplotlib
import wandb

# Log key metrics every episode
wandb.log({
    'episode_reward': episode_reward,
    'episode_length': episode_length,
    'epsilon': current_epsilon,
    'q_loss': current_loss,
    'reward_ma_100': reward_ma_100
})

# Log detailed metrics every N episodes
if episode % 100 == 0:
    wandb.log({
        'q_mean': q_mean,
        'q_std': q_std,
        'td_error_mean': td_error_mean,
        'gradient_norm': gradient_norm,
        'success_rate': success_rate
    })
```

### Alert Systems
```python
# Set up alerts for critical issues
def check_training_health():
    alerts = []
    
    # Reward collapse
    if reward_ma_100 < 0.5 * max(episode_rewards):
        alerts.append("Potential reward collapse detected")
    
    # Loss explosion
    if current_loss > 10 * loss_ma:
        alerts.append("Loss spike detected - check learning rate")
    
    # Gradient explosion
    if gradient_norm > 100:
        alerts.append("Gradient explosion - reduce learning rate")
    
    # No exploration
    if current_epsilon < 0.001 and episode < 0.5 * total_episodes:
        alerts.append("Exploration ended too early")
    
    return alerts
```

## Visualization Best Practices

### Essential Plots
1. **Episode Reward vs Time** (with moving averages)
2. **Loss vs Training Steps** (with moving average)
3. **Q-Value Distribution** (histogram, updated periodically)
4. **Epsilon Decay Schedule** (actual vs planned)
5. **Action Distribution** (bar chart, updated periodically)

### Advanced Analysis
1. **TD Error Heatmaps** (for state-action pairs)
2. **Q-Value Evolution** (same states over time)
3. **Gradient Flow Analysis** (layer-wise gradient norms)
4. **Correlation Analysis** (between different metrics)

## Monitoring Checklist

### Every Episode:
- ✅ Episode reward and length
- ✅ Current epsilon value
- ✅ Moving average rewards

### Every Training Step:
- ✅ Q-loss value
- ✅ Gradient norms
- ✅ TD errors

### Every 100 Episodes:
- ✅ Q-value statistics
- ✅ Action distribution
- ✅ Success rate (if applicable)
- ✅ Buffer statistics

### Every 1000 Episodes:
- ✅ Full model evaluation
- ✅ Policy visualization
- ✅ Hyperparameter effectiveness review

## Red Flags to Watch For

🚨 **Immediate Action Required:**
- Loss suddenly increases by >10x
- Gradient norms >100
- Reward drops by >50% suddenly
- All actions converge to single action early in training

⚠️ **Warning Signs:**
- Reward plateau for >1000 episodes
- Loss oscillates wildly
- Q-values grow without bound
- Very low action diversity

✅ **Healthy Training Indicators:**
- Smooth reward increase
- Stable, decreasing loss
- Moderate gradient norms (0.1-10)
- Balanced exploration-exploitation transition

The key is to monitor these metrics continuously and understand their relationships. Good monitoring allows you to catch issues early and make informed hyperparameter adjustments!
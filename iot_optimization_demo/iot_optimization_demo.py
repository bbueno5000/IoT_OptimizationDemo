"""
DOCSTRING
"""
import collections
import sys

class BlackJack:
    """
    DOCSTRING
    """
    def __init__(self):
        env = BlackjackEnv()

    def __call__(self):
        for i_episode in range(20):
            observation = env.reset()
            for t in range(100):
                print_observation(observation)
                action = strategy(observation)
                print("Taking action: {}".format(["Stick", "Hit"][action]))
                observation, reward, done, _ = env.step(action)
                if done:
                    print_observation(observation)
                    print("Game end. Reward: {}\n".format(float(reward)))
                    break

    def print_observation(self, observation):
        """
        DOCSTRING
        """
        score, dealer_score, usable_ace = observation
        log = 'Player Score: {} (Usable Ace: {}), Dealer Score: {}'
        print(log.format(score, usable_ace, dealer_score))

    def strategy(self, observation):
        """
        DOCSTRING
        """
        score, dealer_score, usable_ace = observation
        return 0 if score >= 20 else 1

class MonteCarloPrediction:
    """
    DOCSTRING
    """
    def __init__(self):
        env = BlackjackEnv()
    
    def __call__(self):
        V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
        plotting.plot_value_function(V_10k, title="10,000 Steps")
        V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
        plotting.plot_value_function(V_500k, title="500,000 Steps")

    def mc_prediction(self, policy, env, num_episodes, discount_factor=1.0):
        """
        Monte Carlo prediction algorithm.
        Calculates the value function for a given policy using sampling.
    
        Args:
            policy: A function that maps an observation to action probabilities.
            env: OpenAI gym environment.
            num_episodes: Number of episodes to sample.
            discount_factor: Gamma discount factor.
    
        Returns:
            A dictionary that maps from state -> value.
            The state is a tuple and the value is a float.
        """
        returns_sum = collections.defaultdict(float)
        returns_count = collections.defaultdict(float)
        V = collections.defaultdict(float)
        for i_episode in range(1, num_episodes + 1):
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()
            episode = list()
            state = env.reset()
            for t in range(100):
                action = policy(state)
                next_state, reward, done, _ = env.step(action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state
            states_in_episode = set([tuple(x[0]) for x in episode])
            for state in states_in_episode:
                first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
                G = sum(
                    [x[2] * (discount_factor**i) for i, x in enumerate(
                        episode[first_occurence_idx:])])
                returns_sum[state] += G
                returns_count[state] += 1.0
                V[state] = returns_sum[state] / returns_count[state]
        return V

    def sample_policy(self, observation):
        """
        A policy that sticks if the player score is >= 20 and hits otherwise.
        """
        score, dealer_score, usable_ace = observation
        return 0 if score >= 20 else 1

class SmartHome:
    """
    DOCSTRING
    """
    def __init__(self):
        env = SmartHomeEnv()

    def __call__(self):
        V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
        plotting.plot_value_function(V_10k, title='10,000 Steps')
        V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
        plotting.plot_value_function(V_500k, title='500,000 Steps')

    def mc_prediction(self, policy, env, num_episodes, discount_factor=1.0):
        """
        Monte Carlo prediction algorithm.
        Calculates the value function for a given policy using sampling.
        
        Args:
            policy: A function that maps an observation to action probabilities.
            env: Smart Home environment.
            num_episodes: Number of episodes to sample.
            discount_factor: Gamma discount factor.
        
            Returns:
            A dictionary that maps from state -> value.
            The state is a tuple and the value is a float.
        """
        # init, observation: (Cooling Demands, electricity prices, electricity consumption)
        observation = env.reset()
        returns_sum = collections.defaultdict(float)
        returns_count = collections.defaultdict(float)
        V = collections.defaultdict(float)
        memory = list()
        for i in range(num_episodes):
            # take an action
            action = policy(observation)
            observation, reward, done, _ = env.step(action)
            memory.append((observation, reward, done))
            if done:
                observation = env.reset()
                returns_sum, returns_count = update(
                    returns_sum,
                    returns_count,
                    memory,
                    discount_factor=discount_factor)
                memory = []
        for i in returns_sum.keys():
            V[i] = returns_sum[i] / returns_count[i]
        return V

    def sample_policy(self, observation):
        """
        A policy that sticks if the player score is > 20 and hits otherwise.
        """
        score, dealer_score, usable_ace = observation
        return 0 if score >= 20 else 1
    
    def update(self, dict_sum, dict_count, tra, discount_factor):
        """
        CaLculate the reward of a given trajectory and record them in dict_sum, dict_count.

        Args:
            dict_sum: the dictionary that record the sum of reward of different states.
            dict_count: the dictionary that record the count of different states.
            tra: trajectory.

        Returns:
            dict_sum: updated
            dict_count: updated
        """
        value = 0.0
        for t in tra[::-1]:
            observation, reward, _ = t
            value = value * discount_factor + reward
            dict_sum[observation] += value
            dict_count[observation] += 1
        return dict_sum, dict_count

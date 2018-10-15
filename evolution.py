import numpy as np
from evostra import EvolutionStrategy as Eee



class EvolutionStrategy:

    def __init__(self, model, weights, env):
        self.model = model
        self.weights = weights
        self.POPULATION_SIZE = 20
        self.SIGMA = 0.1
        self.LEARNING_RATE = 0.01
        self.decay = 0.999
        self.env = env

        self.es = Eee(self.weights, self.__get_reward)

    def __update_weights(self):
        pass

    def __get_population_rewards(self, population_weights):
        # self.env.step
        solution = 0 # target
        rewards = []
        for w in population_weights:
            reward = self.__get_reward(w)
            rewards.append(reward)

        normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)

        return normalized_rewards, rewards

    def __get_reward(self, current_weights):
        self.model.set_weights(current_weights)

        rewards = 0
        episodes = 1
        for _ in range(episodes):
            done = False
            obs = self.env.reset()
            obs = obs.reshape([1, 4])
            while not done:
                prediction = self.model.predict(obs)
                prediction = np.argmax(prediction)
                obs, reward, done, _ = self.env.step(prediction)
                # self.env.render()
                obs = obs.reshape([1, 4])
                rewards += reward

        return rewards/episodes

    def __generate_population_weights(self):
        population_weights = []
        for i in range(self.POPULATION_SIZE):
            weights_jitter = []
            for w in self.weights:
                weights_jitter.append(np.random.randn(*w.shape) * self.SIGMA)
            current_weights = self.weights + weights_jitter
            population_weights.append(current_weights)
        return population_weights

    def update(self):
        self.es.run(600, print_step=1)
        # update the weights

        # generate population weights
        population_weights = self.__generate_population_weights()
        population_norm_rewards, rewards = self.__get_population_rewards(population_weights)

        # update self.weights
        for index, w in enumerate(self.weights):
            current_weight = np.array([population_weight[index] for population_weight in population_weights])
            obj_func = np.dot(current_weight.T, population_norm_rewards).T
            self.weights[index] = w + self.LEARNING_RATE / (self.SIGMA * self.POPULATION_SIZE) * obj_func
            self.LEARNING_RATE = self.LEARNING_RATE * self.decay

        return np.max(rewards)


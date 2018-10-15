import gym
import keras
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.models import Model

from evolution import EvolutionStrategy

env = gym.make('CartPole-v0')
env.reset()

input_layer = Input(shape=(4,))
layer_1 = Dense(4, activation='relu')(input_layer)
output_layer = Dense(env.action_space.n, activation='softmax')(layer_1)
model = Model(input_layer, output_layer)
model.compile(Adam(), 'mse', metrics=['accuracy'])

evolution_strategy = EvolutionStrategy(model, model.get_weights(), env)

rewards = 0
for _ in range(1000):
   current_rewards = evolution_strategy.update()

   if current_rewards > rewards:
       print(current_rewards)
import tensorflow
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

import os

#os.environ['KMP_DUPLICATE_LIB_OK']='True'
ray.init()

# ray config
config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 1

# train on cartpole
trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
result = trainer.train()
pretty_print(result)

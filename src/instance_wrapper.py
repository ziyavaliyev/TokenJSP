import gymnasium as gym
import numpy as np

class InstanceSampler(gym.Wrapper):
    def __init__(self, env: gym.Env, instances: list[np.ndarray]):
        super().__init__(env)
        self.instances = [np.asarray(x) for x in instances]
        self._order = np.arange(len(self.instances))
        self._i = 0

    def reset(self, **kwargs):
        # optionally we can reshuffle when we completed one full pass
        if self._i % len(self.instances) == 0:
            self._i = 0
        inst = self.instances[self._order[self._i]]
        print(self._i)
        self._i += 1
        self.env.unwrapped.load_instance(inst)
        return self.env.reset(**kwargs)
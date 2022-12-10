from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CustomEvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, eval_freq=0, verbose=0):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            last_mean_reward = 0
            aoi_records = self.training_env.get_attr('AoI_record')
            num_record_each_env = 20 // len(aoi_records)
            for aoi_record in aoi_records:
                last_mean_reward -= np.sum(aoi_record[- num_record_each_env:])
            last_mean_reward /= num_record_each_env * len(aoi_records)
            print(f'''[{self.n_calls}/{self.num_timesteps}={np.round(self.n_calls/self.num_timesteps*100, 2)}%] Mean Weighted AoI for last 20 rounds in training: {last_mean_reward}''')

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
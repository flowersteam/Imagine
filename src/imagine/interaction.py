from collections import deque
import pickle
import numpy as np
from mpi4py import MPI

class RolloutWorker:
    """Rollout worker generates experience by interacting with one or many environments.

    Args:
        make_env (function): a factory function that creates a new instance of the environment
            when called
        policy (object): the policy that is used to act
        T (int): number of timesteps in an episode
        eval_bool (bool): whether it is an evaluator rollout worker or not
        rollout_batch_size (int): the number of parallel rollouts that should be used
        exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
            current policy without any exploration
        use_target_net (boolean): whether or not to use the target net for rollouts
        compute_Q (boolean): whether or not to compute the Q values alongside the actions
        noise_eps (float): scale of the additive Gaussian noise
        random_eps (float): probability of selecting a completely random action
        history_len (int): length of history for statistics smoothing
        render (boolean): whether or not to render the rollouts
    """
    def __init__(self,
                 make_env,
                 policy,
                 T,
                 eval_bool,
                 reward_function,
                 rollout_batch_size=1,
                 exploit=False,
                 use_target_net=False,
                 compute_Q=False,
                 noise_eps=0,
                 random_eps=0,
                 history_len=100,
                 render=False,
                 save_obs=False,
                 params={},
                 **kwargs):

        self.T = T
        self.policy = policy
        self.reward_function = reward_function
        self.eval = eval_bool
        self.rollout_batch_size = rollout_batch_size
        self.exploit = exploit
        self.use_target_net = use_target_net
        self.compute_Q = compute_Q
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.history_len = history_len
        self.render = render
        self.save_obs = save_obs
        self.params = params.copy()
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.env = make_env()

        assert self.T > 0

        self.Q_history = deque(maxlen=history_len)


    def generate_rollouts(self, exploit, imagined, goals_str, goals_encodings, goals_ids):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.exploit = exploit
        self.imagined = imagined
        assert len(goals_str) == self.rollout_batch_size == len(goals_encodings) == len(goals_ids)
        episodes = []

        for i in range(self.rollout_batch_size):

            # Reset the environment
            env_seed = np.random.randint(int(1e6))
            self.env.seed(env_seed)
            self.env.reset()
            initial_o = self.env.unwrapped.reset_with_goal(goals_str[i])
            o = initial_o.copy()
            Qs = []
            obs = [o.copy()]
            acts = []

            goal = goals_encodings[i].copy()
            # Run a rollout
            for t in range(self.T):
                # Get next action from policy
                policy_output = self.policy.get_actions(o.copy(),
                                                        goal,
                                                        compute_Q=self.compute_Q,
                                                        noise_eps=self.noise_eps if not self.exploit else 0.,
                                                        random_eps=self.random_eps if not self.exploit else 0.,
                                                        use_target_net=self.use_target_net
                                                        )

                if self.compute_Q:
                    u, Q = policy_output
                    Qs.append(Q)
                else:
                    u = policy_output

                # Env step
                o, _, _, _ = self.env.step(u)

                if self.render:
                    self.env.render()

                obs.append(o.copy())
                acts.append(u.copy())

            episode = dict(obs=np.array(obs),
                           acts=np.array(acts),
                           g_encoding=goals_encodings[i],
                           g_id=goals_ids[i],
                           g_str=goals_str[i],
                           exploit=self.exploit,
                           imagined=self.imagined
                           )
            episodes.append(episode)

        # stats
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))

        return episodes


    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        self.policy.save_model(path)

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        self.env.seed(seed + 1000 * self.rank)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.Q_history.clear()
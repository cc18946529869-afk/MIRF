import numpy as np
from stable_baselines3.ppo.ppo import PPO
# from . import register_algorithm
from .environments.env import LandscapeEnv
import flexs
from flexs import baselines


# @register_algorithm("dirichlet_ppo")
class MuSearch(flexs.Explorer):
    """ PPO-based Protein Optimization with Dirichlet Sampling. """
    def __init__(self, args, oracle_model, alphabet, starting_sequence, sequences_batch_size, model_queries_per_batch, rounds, log_file=None, update_frequency=3):
        model = oracle_model
        name = "MuSearch"
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )
        self.oracle_model = oracle_model
        self.alphabet = alphabet
        self.starting_sequence = starting_sequence
        self.score_threshold = args.score_threshold
        self.num_trajs_per_update = model_queries_per_batch // update_frequency
        self.total_timesteps = model_queries_per_batch * args.horizon
        self.horizon = args.horizon

    def propose_sequences(self, input_sequences):
        print("-----------------MuSearch-----------------")
        print(f"input_sequences", input_sequences)
        # input_sequence is a pandas dataframe
        # There is a column named model_score in the dataframe
        # Rank the sequences based on the model_score
        # Set the top sequences as the starting sequences for the PPO
        starting_sequence = input_sequences.sort_values(by="true_score", ascending=False).head(1).sequence.values[0]
        self.starting_sequence = starting_sequence
        env_fn = lambda :LandscapeEnv(self.oracle_model, self.alphabet, self.horizon, self.starting_sequence)
        env = env_fn()
        example_env = env_fn()
        n_steps= self.horizon * self.num_trajs_per_update
        rl_agent = PPO(["MlpPolicy", "MlpPolicy"], env, batch_size=512, verbose=1, n_steps=n_steps, for_protein=True, score_threshold=self.score_threshold, tensorboard_log="./logs",
        policy_kwargs={
                        'sub_step_observation_spaces1': example_env.observation_space1,
                        'sub_step_observation_spaces2': example_env.observation_space2,
                        'sub_step_action_spaces1': example_env.action_space1,
                        'sub_step_action_spaces2': example_env.action_space2,
                    })
        print("-----------------MuSearch-----------------")
        print(f"num_envs", rl_agent.env.num_envs)
        exploration_candidate_pool, sampling_candidate_pool = rl_agent.learn(self.total_timesteps)
        print("Total sample sequence num: ", len(exploration_candidate_pool))

        all_explored_sequences = {}
        filtered_sequences = {}

        for arr, score, embedding, ensemble_uncertainty in exploration_candidate_pool:
            candidate = example_env.array2str(arr)
            if score > self.score_threshold:
                filtered_sequences[candidate] = score
            all_explored_sequences[candidate] = score

        print("Unique sequences num during training: ", len(all_explored_sequences))
        print("Unique sequences num after filtering: ", len(filtered_sequences))
        
        sequences = list(filtered_sequences.keys())
        scores = list(filtered_sequences.values())
        ranked_idx = np.argsort(scores)[: -self.sequences_batch_size : -1]
        sequences = np.array(sequences)[ranked_idx]
        scores = np.array(scores)[ranked_idx]
        return sequences, scores
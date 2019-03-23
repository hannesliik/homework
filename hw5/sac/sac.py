import time

import tensorflow as tf


class SAC:
    """Soft Actor-Critic (SAC)
    Original code from Tuomas Haarnoja, Soroush Nasiriany, and Aurick Zhou for CS294-112 Fall 2018

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," ICML 2018.
    """

    def __init__(self,
                 alpha=1.0,
                 batch_size=256,
                 discount=0.99,
                 epoch_length=1000,
                 learning_rate=3e-3,
                 reparameterize=False,
                 tau=0.01,
                 **kwargs):
        """
        Args:
        """

        self._alpha = alpha
        self._batch_size = batch_size
        self._discount = discount
        self._epoch_length = epoch_length
        self._learning_rate = learning_rate
        self._reparameterize = reparameterize
        self._tau = tau

        self._training_ops = []

    def build(self, env, policy, q_function, q_function2, value_function,
              target_value_function):

        self._create_placeholders(env)

        policy_loss = self._policy_loss_for(policy, q_function, q_function2, value_function)
        value_function_loss = self._value_function_loss_for(
            policy, q_function, q_function2, value_function)
        q_function_loss = self._q_function_loss_for(q_function,
                                                    target_value_function)
        if q_function2 is not None:
            q_function2_loss = self._q_function_loss_for(q_function2,
                                                         target_value_function)

        optimizer = tf.train.AdamOptimizer(
            self._learning_rate, name='optimizer')
        policy_training_op = optimizer.minimize(
            loss=policy_loss, var_list=policy.trainable_variables)
        value_training_op = optimizer.minimize(
            loss=value_function_loss,
            var_list=value_function.trainable_variables)
        q_function_training_op = optimizer.minimize(
            loss=q_function_loss, var_list=q_function.trainable_variables)
        if q_function2 is not None:
            q_function2_training_op = optimizer.minimize(
                loss=q_function2_loss, var_list=q_function2.trainable_variables)

        self._training_ops = [
            policy_training_op, value_training_op, q_function_training_op
        ]
        if q_function2 is not None:
            self._training_ops += [q_function2_training_op]
        self._target_update_ops = self._create_target_update(
            source=value_function, target=target_value_function)

        tf.get_default_session().run(tf.global_variables_initializer())

    def _create_placeholders(self, env):
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='observation',
        )
        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, action_dim),
            name='actions',
        )
        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None,),
            name='rewards',
        )
        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None,),
            name='terminals',
        )

    def _policy_loss_for(self, policy, q_function, q_function2, value_function):
        actions, log_pis = policy(self._observations_ph)
        if q_function2 is not None:
            Q_sa = tf.minimum(q_function((self._observations_ph, actions)), q_function2((self._observations_ph, actions)))
        else:
            Q_sa = q_function((self._observations_ph, actions))
        if self._reparameterize:
            # reparam trick
            #### Problem 1.3.B
            ### YOUR CODE HERE
            return tf.reduce_mean(self._alpha * log_pis - tf.log(Q_sa))
        else:
            # REINFORCE
            ## Problem 1.3.A
            ### YOUR CODE HERE

            V_s = value_function(self._observations_ph)
            policy_loss = log_pis * (tf.stop_gradient(self._alpha * log_pis - Q_sa + V_s))
            policy_loss = tf.reduce_mean(policy_loss)
            return policy_loss

    def _value_function_loss_for(self, policy, q_function, q_function2, value_function):
        ### Problem 1.2.A
        ### YOUR CODE HERE
        V_s = value_function(self._observations_ph)
        actions, log_pis = policy(self._observations_ph)
        if q_function2 is not None:
            Q_s = tf.minimum(q_function((self._observations_ph, actions)),
                              q_function2((self._observations_ph, actions)))
        else:
            Q_s = q_function((self._observations_ph, actions))
        #probs = tf.exp(log_pis)
        #inner_expectation = tf.stop_gradient(tf.reduce_sum(probs * (Q_s - self._alpha * log_pis), axis=1))
        inner_expectation = tf.stop_gradient(Q_s - self._alpha * log_pis)
        return tf.losses.mean_squared_error(inner_expectation, V_s) # Expectation is basically mean (over batch), right?

    def _q_function_loss_for(self, q_function, target_value_function):
        ### Problem 1.1.A
        ### YOUR CODE HERE
        #q_values = tf.log(tf.reduce_sum(tf.exp(q_function((self._observations_ph, self._actions_ph))), axis=1)) #soft max
        q_values = q_function((self._observations_ph, self._actions_ph))
        targets = self._rewards_ph + self._discount * target_value_function(self._next_observations_ph) * (
                    1 - self._terminals_ph)
        print(q_values.shape, targets.shape)
        loss = tf.losses.mean_squared_error(tf.stop_gradient(targets), q_values)
        # Maybe shouldn't stop gradients?
        return loss

    def _create_target_update(self, source, target):
        """Create tensorflow operations for updating target value function."""

        return [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target.trainable_variables, source.
                                      trainable_variables)
        ]

    def train(self, sampler, n_epochs=1000):
        """Return a generator that performs RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._start = time.time()
        for epoch in range(n_epochs):
            for t in range(self._epoch_length):
                sampler.sample()

                batch = sampler.random_batch(self._batch_size)
                feed_dict = {
                    self._observations_ph: batch['observations'],
                    self._actions_ph: batch['actions'],
                    self._next_observations_ph: batch['next_observations'],
                    self._rewards_ph: batch['rewards'],
                    self._terminals_ph: batch['terminals'],
                }
                tf.get_default_session().run(self._training_ops, feed_dict)
                tf.get_default_session().run(self._target_update_ops)

            yield epoch

    def get_statistics(self):
        statistics = {
            'Time': time.time() - self._start,
            'TimestepsThisBatch': self._epoch_length,
        }

        return statistics

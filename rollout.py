
def collect_trajectory(pi: Policy, env=env) -> Trajectory:

    # to avoid python's list append behavior
    trajectory = Trajectory()
    if trajectory.states is None:
        trajectory = trajectory._replace(states=[])
    if trajectory.actions is None:
        trajectory = trajectory._replace(actions=[])
    if trajectory.action_probs is None:
        trajectory = trajectory._replace(action_probs=[])
    if trajectory.rewards is None:
        trajectory = trajectory._replace(rewards=[])

    done = False

    state = env.reset()
    trajectory.states.append(state)

    while not done:
        if args.render:
            env.render()
        # need to wrap in np.array([]) for `predict` to work
        # `predict` returns 2D array with single 1D array that we need to extract
        action_probs = pi.predict(np.array([state]), batch_size=1)[0]
        action = np.random.choice(np.arange(NUM_ACTIONS), p=action_probs)
        state, reward, done, _ = env.step(action)

        trajectory.states.append(state)
        trajectory.actions.append(action)
        trajectory.action_probs.append(action_probs[action])
        trajectory.rewards.append(reward)

    # Cast elements of tuple to Numpy arrays.
    # assign 0 reward to final state
    trajectory.rewards.append(0.0)
    trajectory = Trajectory(
        np.array(trajectory.states),
        np.array(trajectory.actions),
        np.array(trajectory.action_probs),
        np.array(trajectory.rewards), )

    return trajectory


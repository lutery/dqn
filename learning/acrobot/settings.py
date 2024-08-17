params = {
        'env_name':         "Acrobot-v1",
        'stop_reward':      -100.0,
        'run_name':         'acrobot',
        'replay_size':      10 ** 6,
        'replay_initial':   20000,
        'target_net_sync':  5000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    5e-4,
        'gamma':            0.99,
        'batch_size':       128
    }
from gym.envs.registration import register

register(
    id='ENVTEST-v1',
    entry_point='custom_gym.myenv.env_test_1:EnvTest1',
)

register(
    id='ENVTEST-v2',
    entry_point='custom_gym.myenv.env_test_2:EnvTest2',
)

register(
    id='ENVTEST-v3',
    entry_point='custom_gym.myenv.env_test_multi_agent:EnvTestMultiAgent',
)

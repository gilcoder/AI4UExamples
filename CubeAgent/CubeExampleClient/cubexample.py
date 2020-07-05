from ai4u.core import RemoteEnv

env = RemoteEnv()
env.open()

env.step('tx', 5)

env.close()
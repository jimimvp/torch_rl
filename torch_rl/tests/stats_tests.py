from torch_rl.stats import RLTrainingStats
import matplotlib.pyplot as plt

df = RLTrainingStats.load("/disk/users/vlasteli/no_backup/Projects/torch_rl/examples/training_stats")

total_rows = df.count()

print(df.keys())
df.plot(y='mvavg_reward')
plt.show()
import matplotlib.pyplot as plt

tokens = ["six", "drinking", "cup", "20", "productivity", "researcher", "two", "scientific", "day", "increase"]
weights = [1]*10

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(tokens, weights)
plt.xticks(rotation=45, ha='right')
fig.savefig('test_plot.png')

"""This is a small script to plot a zero-shot classification output nicely
for the front page of the paper. Needs hard-coded class names and confidences."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SCORES = np.array([0.0031, 0.5265, 0.0087, 0.2124, 0.0699, 0.1794])
NAMES = np.array(["animal", "cyclist", "person", "truck", "bus", "car"])
ORDER = [0, 2, 1, 5, 4, 3]

# Make a very very pretty column plot using seaborn
sns.set_style("whitegrid")
sns.set_context("poster")
f, ax = plt.subplots(figsize=(12, 8))
# Use a beautiful color palette that fits in an academic paper
palette = sns.color_palette("husl", len(SCORES))
sns.barplot(x=SCORES[ORDER], y=NAMES[ORDER], palette=palette)
# set ticks to 0, 0.2, 0.4, 0.6, 0.8, 1.0
ax.set_xticks(np.arange(0, 1.1, 0.2))
# expand the plot to the left to avoid truncating the labels
plt.subplots_adjust(left=0.4)
# larger font for the labels
plt.tick_params(axis="y", labelsize=45)
plt.tick_params(axis="x", labelsize=30)
plt.savefig("classification_plot.png", dpi=300)
plt.show()

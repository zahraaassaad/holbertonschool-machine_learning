#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fig, ax = plt.subplots()

columns = ('Farrah', 'Fred', 'Felicia')
rows = ('apples', 'bananas', 'oranges', 'peaches')
colors = ('r', 'y', '#ff8000', '#ffe5b4')
index = np.zeros(3)
bar_width = 0.5

for i, n_fruits in enumerate(fruit):
        ax.bar(
            columns,
            n_fruits,
            bar_width,
            bottom=index,
            label=rows[i],
            color=colors[i]
        )
        index += n_fruits
ax.set_ylabel('Quantity of Fruit')
ax.set_ylim(0, 80)
ax.set_title('Number of Fruit per Person')
ax.legend()

plt.show()

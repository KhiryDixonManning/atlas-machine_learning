#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def bars():
    np.random.seed(5)

    # fruit matrix: rows are fruits (apples, bananas, oranges, peaches)
    # columns are people (Farrah, Fred, Felicia)
    fruit = np.random.randint(0, 20, (4, 3))

    # Fruit names for the rows (apples, bananas, oranges, peaches)
    fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']

    # Colors for each fruit type
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Plot the stacked bar chart
    plt.figure(figsize=(6.4, 4.8))  # Optional: set the figure size

    # Create a stacked bar chart
    # Bottom argument ensures stacking of bars
    plt.bar([0, 1, 2], fruit[0], color=colors[0], label=fruit_names[0], width=0.5)  # Apples
    plt.bar([0, 1, 2], fruit[1], bottom=fruit[0], color=colors[1], label=fruit_names[1], width=0.5)  # Bananas
    plt.bar([0, 1, 2], fruit[2], bottom=fruit[0] + fruit[1], color=colors[2], label=fruit_names[2],
            width=0.5)  # Oranges
    plt.bar([0, 1, 2], fruit[3], bottom=fruit[0] + fruit[1] + fruit[2], color=colors[3], label=fruit_names[3],
            width=0.5)  # Peaches

    # Labels and title
    plt.xlabel('Person')  # x-axis label
    plt.ylabel('Quantity of Fruit')  # y-axis label
    plt.title('Number of Fruit per Person')  # Title of the plot

    # Set y-axis range and ticks
    plt.ylim(0, 80)  # y-axis range from 0 to 80
    plt.yticks(range(0, 81, 10))  # y-axis ticks every 10 units

    # Set x-axis labels (names of the people)
    plt.xticks([0, 1, 2], ['Farrah', 'Fred', 'Felicia'])  # x-axis ticks with names

    # Add legend
    plt.legend(title="Fruit Types")

    # Show the plot
    plt.show()

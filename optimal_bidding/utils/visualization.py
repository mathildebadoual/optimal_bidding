import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Plotter():
    def __init__(self):
        self.plots_waiting = []

    def plot(self, to_plot, title, legend, n_data):
        plt.figure(figsize=(10, 5))
        for i, element in enumerate(to_plot):
            plt.plot(element[0][-n_data:], element[1][-n_data:], label=legend[i])
        plt.legend()
        plt.title(title)
        plt.grid()
        file_name = '%s.png' % title
        plt.savefig(file_name)

    def callback(self, to_plot, title, legend):
        self.plots_waiting.append((to_plot, title, legend))

    def plot_all(self):
        for plots in self.plots_waiting:
            self.plot(plots[0], plots[1], plots[2])
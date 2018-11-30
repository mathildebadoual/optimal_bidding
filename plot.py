import argparse
import pickle

from utils.visualization import Plotter


def get_plot_input(save_dict, values):
    to_plot = []
    for value in values:
        to_plot.append((save_dict["date"], save_dict[value]))

    legends = values
    return to_plot, legends


def main():
    # ************ ARGPARSE ************
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str,
                        help='Path of the result file.')
    parser.add_argument('--value', '-v', type=str, default="reward",
                        help='Value of the saved_dict to print')
    args = parser.parse_args()

    pickle_file = args.file
    value = args.value
    title = args.value
    # ************ LOAD DATA ************
    with open(pickle_file, "rb") as file:
        save_dict = pickle.load(file)
    values = save_dict[value]

    # ************ PLOT ************
    to_plot, legends = get_plot_input(save_dict, values)

    plotter = Plotter()
    plotter.plot(to_plot, title, legends, 24)

if __name__ == '__main__':
    main()
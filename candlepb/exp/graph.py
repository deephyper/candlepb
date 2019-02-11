import json
import sys

from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem, SubmenuItem, CommandItem

import matplotlib.pyplot as plt

## Utils
def get_data_from(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def moving_average(data_list, window_size=10):
    res_list = list()
    for i in range(len(data_list) - window_size):
            res_list.append(sum(data_list[i:i+window_size])/window_size)
    return res_list


## Menu
def go_menu():
    from consolemenu import SelectionMenu

    plot_funcs_names = list(filter(lambda e: 'plot' in e, globals()))

    selection_list= [' '.join(e.split('_')) for e in plot_funcs_names]

    menu = SelectionMenu(selection_list, "Select an option")

    menu.show()
    menu.join()

    selection = menu.selected_option
    print('=> ', selection_list[0])
    return globals()[plot_funcs_names[selection]] # function

## Main
def main():
    path = sys.argv[1]
    func = go_menu()
    func(path=path)

## Ploting ##
# each plot function has to start with 'plot_'

def plot_raw_rewards_with_moving_average(path):
    data = get_data_from(path)

    plt.title(f'Raw rewards with moving average: {path.split("/")[-1]}')
    plt.ylabel('Reward')
    plt.xlabel('Number of moodels sampled')

    # plt.ylim(0, 115)

    plt.plot(data['raw_rewards'], alpha=0.5, label='raw')

    w_sizes = [10, 100, 1000]
    for ws in w_sizes:
        avr = moving_average(data_list=data['raw_rewards'], window_size=ws)
        plt.plot(avr, alpha=0.5, label=f'window={ws}')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
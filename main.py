import argparse
import json

import matplotlib.pyplot as plt


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description='An implementation of the Distributional Policy Optimization paper.',
    )
    parser.add_argument(
        '-f', '--results-file', type=str, default='results/results.txt'
    )
    parser.add_argument(
        '-d', '--data-key', type=str, default='eval_rewards'
    )
    parser.add_argument(
        '-y', '--dependent-key', type=str, default='average_eval_reward'
    )
    parser.add_argument(
        '-x', '--independent-key', type=str, default='train_steps'
    )
    parser.add_argument(
        '-t', '--title', type=str, default='Results from ML Run'
    )
    parser.add_argument(
        '-xa', '--x-axis', type=str, default='Training Steps'
    )
    parser.add_argument(
        '-ya', '--y-axis', type=str, default='Rewards'
    )
    parser.add_argument(
        '-sn', '--save-name', type=str, default='results_graph.png'
    )
    parser.add_argument(
        '-s', '--save', default=False, action='store_true'
    )
    parser.add_argument(
        '-yl', '--y-limit', type=str, default=None, help='List of two elements indicating bottom and top values for y axis ([bottom, top]).'
    )
    return parser


def main():
    args = create_argument_parser().parse_args()
    results_string = ''
    with open(args.results_file, 'r') as file:
        results_string = file.read()
    results_dict = json.loads(results_string)
    if args.data_key:
        results_list = results_dict[args.data_key]
    else:
        results_list = results_dict
    result_tuples = list(
        map(
            lambda data_dict: [data_dict[args.independent_key], data_dict[args.dependent_key]],
            results_list
        )
    )
    max_value = max(map(lambda data_dict: data_dict[args.dependent_key], results_list))
    last_ten = results_list[-10:]
    average_last_ten = sum(map(lambda data_dict: data_dict[args.dependent_key], last_ten))
    average_last_ten = average_last_ten/len(last_ten)
    print('Max value achieved:', max_value)
    print('Average value for the last ten steps:', average_last_ten)
    plt.plot(*zip(*result_tuples))
    plt.title(args.title)
    plt.xlabel(args.x_axis)
    plt.ylabel(args.y_axis)
    if args.y_limit:
        y_lim = json.loads(args.y_limit)
        plt.ylim(y_lim)
    if args.save:
        plt.savefig(args.save_name)
    else:
        plt.show()


if __name__ == '__main__':
    main()

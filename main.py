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
    print('Max value achieved:', max_value)
    plt.plot(*zip(*result_tuples))
    plt.title(args.title)
    plt.xlabel(args.x_axis)
    plt.ylabel(args.y_axis)
    if args.save:
        plt.savefig(args.save_name)
    else:
        plt.show()


if __name__ == '__main__':
    main()

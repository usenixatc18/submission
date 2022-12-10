#! /usr/bin/env python3
from tempfile import NamedTemporaryFile
import tarfile
import click
import time
import json
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt

COLORS =  ['r', 'g', 'k', 'm', 'y', 'b', 'c']
MARKERS = ['d', '^', '*', 'x', '+', 's', 'o']
STYLE_MAP = dict()
def get_line_style(name):
    if name not in STYLE_MAP:
        STYLE_MAP[name] = (COLORS[len(STYLE_MAP)], MARKERS[len(STYLE_MAP)])
    return STYLE_MAP[name]

def mean_reward(stats):
    return np.mean(np.array(stats['result']['rewards']), 0) 


def is_learning(s):
    return len(s['experiment']['policies']) > 1


def plot_avg_reward(stats, plt, optimal_mean_reward, plot_type):
    plt.set_xlabel('Step #')
    plt.set_ylabel('Average reward')

    learning_idx = 0
    for i, s in enumerate(stats):
        ys = mean_reward(s)
        if plot_type == 'regret':
            opt_ys = np.cumsum(optimal_mean_reward)
            ys = np.cumsum(ys)
            ys = opt_ys - ys
        elif plot_type == 'relative_cum_reward':
            opt_ys = np.cumsum(optimal_mean_reward)
            ys = np.cumsum(ys) / opt_ys
        elif plot_type == 'relative_reward':
            ys = ys / optimal_mean_reward

        xs = list(range(1, len(ys) + 1))
        if not is_learning(s):
            name = s['experiment']['policies'][0]['type']
        else:
            name = '{} {}'.format('learning', learning_idx)
            learning_idx += 1

        color, marker = get_line_style(name)
        plt.plot(xs, ys, label=name, color=color, marker=marker)

    plt.legend()


def mean_total_reward(stats): 
    return np.sum(mean_reward(stats))


def get_policies(stats):
    return [p['type'] for p in stats['experiment']['policies']]


def print_stats(name, stats, optimal_mean_reward):
    print(f'Statistics for: {name}')
    mean_reward = mean_total_reward(stats)
    print(f" Average total rewards: {mean_reward}")
    best_reward = sum(optimal_mean_reward)
    print(f" Regret: {(best_reward - mean_reward) * 100 / best_reward}%")

def plot_arms(stats, plt):
    plt.set_xlabel('Step #')
    plt.set_ylabel('Arm\'s percentage')
    plt.set_ylim(0.0, 1.0)

    arms = np.array(stats['result']['arms'])
    indiv_arms = np.sum(arms, axis=0) / np.sum(arms, axis=(0,1))

    xs = list(range(1, arms.shape[2] + 1))

    for i, name in enumerate(get_policies(stats)):
        color, marker = get_line_style(name)
        plt.plot(xs, indiv_arms[i], label=name, color=color, marker=marker)
    plt.legend()


@click.group()
def hello():
    pass


def maybe_tar(output):
    if len(output) == 1 and tarfile.is_tarfile(output[0]):
        with tarfile.open(output[0], 'r:gz') as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
            return tar.getnames()
    else:
        return output


def get_phase_sizes(experiment_info):
    total_steps = experiment_info['simulation']['num_time_steps']
    batch_size = experiment_info['infrastructure']['batch_size']
    stat_batch_size = experiment_info['simulation']['stat_batch_size']

    return [ 
        int(s['num_steps'] / batch_size / stat_batch_size)
        for s in experiment_info['source']['parameters']
    ]


def create_phase_masks(phase_sizes, total_size):
    start_idx = 0
    result = []
    for ps in phase_sizes:
        phase_mask = np.zeros(sum(phase_sizes))
        phase_mask[start_idx:start_idx + ps] = 1
        result.append(np.concatenate(
            [phase_mask] * (total_size // sum(phase_sizes))
        ))
        start_idx += ps
    return result



def find_optimal_mean_reward(stats):
    experiment_info = stats[0]['experiment']
    if experiment_info['source']['type'] == 'sequence':
        print('INFO! Guessing that we have a non stationary')
        optimal = np.zeros(len(stats[0]['result']['rewards'][0]))

        phase_sizes = get_phase_sizes(experiment_info)
        phase_masks = create_phase_masks(phase_sizes, len(optimal))
        all_mean_rewards = [mean_reward(s) for s in stats]

        for pm in phase_masks:
            phase_optimal = max(all_mean_rewards, key=lambda mr: sum(mr * pm))
            optimal += phase_optimal * pm
        return optimal
    else:
        optimal = max(stats, key=lambda s: mean_total_reward(s))
        return mean_reward(optimal)


@hello.command()
@click.option('--combined', is_flag=True)
@click.option('--save', type=str, required=False)
@click.option('--plot-type', type=click.Choice(['regret',
    'relative_cum_reward', 'relative_reward']), required=True)
@click.option('--no-arms', is_flag=True)
@click.argument('output', nargs=-1)
def plot(output, save, combined, plot_type, no_arms):
    output = maybe_tar(output)
    stats = []
    for i, o in enumerate(output):
        with open(o) as f:
            stats.append(json.load(f))

    optimal_mean_reward = find_optimal_mean_reward(stats)
    for o, s in zip(output, stats):
        print_stats(o, s, optimal_mean_reward)

    if not combined:
        _, axes = plt.subplots(
            len(output), 1 if no_arms else 2,
            squeeze=False, figsize=(50,10)
        )
        for i, s in enumerate(stats):
            plot_avg_reward([s], axes[i][0], optimal_mean_reward, plot_type)
            if not no_arms:
                plot_arms(s, axes[i][1])
    else:
        learnings = [s for s in stats if is_learning(s)]
        _, axes = plt.subplots(
                1, 1 + (0 if no_arms  else len(learnings)),
                squeeze=False, figsize=(50,10))
        plot_avg_reward(stats, axes[0][0], optimal_mean_reward, plot_type)
        if not no_arms:
            for i, s in enumerate(learnings):
                plot_arms(s, axes[0][1 + i])

    if save is None:
        plt.show()
    else:
        plt.savefig(save)

@hello.command()
@click.argument('output', nargs=-1)
def stats(output):
    output = maybe_tar(output)

    stats = []
    for o in output:
        with open(o) as f:
            stats.append(json.load(f))
    optimal_mean_reward = find_optimal_mean_reward(stats)
    for o, s in zip(output, stats):
        print_stats(o, s, optimal_mean_reward)


@hello.command()
@click.argument("output", type=click.File("r"))
def get_config(output):
    stats = json.load(output)
    print(json.dumps(stats['experiment'], indent=2))


if __name__ == '__main__':
    hello()



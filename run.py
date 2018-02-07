#! /usr/bin/env python3
import os
import asyncio
from concurrent.futures import ProcessPoolExecutor
from os import path
from tempfile import NamedTemporaryFile
from itertools import product
from functools import wraps
from collections import namedtuple 
import subprocess
import tarfile

import click
import json

FixedParams = namedtuple('FixedParams',[
    'num_runs',
    'num_time_steps',
    'num_stat_steps',
    'directory',
    'max_reward',
    'executor',
    'num_processes'
    ])
FIXED_PARAMS = None

@click.group()
@click.option('--num-runs', type=int, required=True)
@click.option('--num-time-steps', type=int, required=True)
@click.option('--num-stat-steps', type=int, required=False)
@click.option('--directory', type=str, default='data', required=False)
@click.option('--max-reward', type=float, required=False)
@click.option('--num-processes', default=1, type=int)
def hello(num_runs, num_time_steps, num_stat_steps, directory, max_reward, 
          num_processes):
    global FIXED_PARAMS
    FIXED_PARAMS = FixedParams(
        num_runs=num_runs, num_time_steps=num_time_steps,
        num_stat_steps=num_stat_steps, directory=directory,
        max_reward=max_reward, num_processes=num_processes,
        executor=ProcessPoolExecutor(max_workers=num_processes))

def current_dir():
    return path.dirname(path.abspath(__file__))


PURE_POLICIES = {
    'vs': {"type": "value/slack"},
    'v': {"type": "value"},
    'd': {"type": "deadline"}
}

PURE_LEARNING = {
    'type': 'constant',
    'parameters': { 'arm_idx': 0 }
}


def calc_num_stat_steps(batch_size):
    num_stat_steps = (FIXED_PARAMS.num_stat_steps if FIXED_PARAMS.num_stat_steps
        is not None else FIXED_PARAMS.num_time_steps / batch_size)
    if (FIXED_PARAMS.num_time_steps / batch_size) % num_stat_steps != 0:
        print("WARN! num_stat_steps does not divide actual number of steps")
    return num_stat_steps


def create_config(batch_size, num_simulated, learning, policies, source):
    if (FIXED_PARAMS.num_time_steps % batch_size) != 0:
        print("WARN! batch_size does not divide actual number of steps")
    num_stat_steps = calc_num_stat_steps(batch_size)
    return {
        'learning': learning,
        'policies': policies,
        'source': source,
        'simulation': {
            'num_time_steps': FIXED_PARAMS.num_time_steps,
            'num_runs': FIXED_PARAMS.num_runs,
            'stat_batch_size': 
                (FIXED_PARAMS.num_time_steps / batch_size) / num_stat_steps
        },
        'infrastructure': {
            'buffer_size': 50,
            'batch_size': batch_size,
            'num_simulated': num_simulated
        },
        'reward': {
            'type': 'weighted_throughput',
            'parameters': None
        }
    }


def create_pure_or_learning(batch_size, num_simulated, algo, source, learning_gen):
    if algo in PURE_POLICIES:
        learning = PURE_LEARNING
        policies = [PURE_POLICIES[algo]]
    else:
        learning = learning_gen(algo)
        policies = list(PURE_POLICIES.values())
    return create_config(batch_size, num_simulated, learning, policies, source)


def enable_scale(func):
    @wraps(func)
    def helper(batch_size, params):
        learning = func(batch_size, params)
        if 'scale' in params:
            assert(FIXED_PARAMS.max_reward is not None)
            return {
                'type': 'scaled',
                'parameters': { 
                    'base': learning, 
                    'factor': 1 / (FIXED_PARAMS.max_reward * batch_size) 
                }
            }
        else: return learning
    return helper


def enable_norm(func):
    @wraps(func)
    def helper(batch_size, params):
        learning = func(batch_size, params)
        if 'norm' in params:
            return {'type': 'normalized',
                    'parameters': { 'base': learning }}
        else:
            return learning
    return helper

def enable_stationary_ucbe(func):
    @wraps(func)
    def helper(batch_size, params):
        learning = func(batch_size, params)
        kw_params = get_kw_params(params)
        if 'ucbe' in kw_params or 'ucbe' in params:
            a = kw_params['ucbe'] if 'ucbe' in kw_params else 2.0
            return {
                'type': 'combined',
                'parameters': {
                    'exploiter': learning,
                    'explorer' : {
                        'type': 'ucb_e',
                        'parameters': {'a': 
                            float(a) * FIXED_PARAMS.num_time_steps / batch_size 
                        }
                    }
                }
            }
        else:
            return learning
    return helper

@enable_norm
@enable_scale
@enable_stationary_ucbe
def stationary_ucb(batch_size, params):
    return {
        "type": "ucb",
        "parameters": { 
            "average": { "type": "keep_all" }, 
            "ksi": 2.0,
            "restricted_exploration": 'rexp' in params
        }
    }


@enable_norm
@enable_scale
@enable_stationary_ucbe
def stationary_greedy(batch_size, params):
    return {
        'type': 'epsilon_greedy',
        'parameters' : {
            'epsilon': float(params[0]),
            'average' : {'type': 'keep_all'},
            'delta': 0.0,
            'sigma': 1.0
        }
    }

@enable_norm
@enable_scale
@enable_stationary_ucbe
def stationary_ucb_tuned(batch_size, params):
    return {
        'type': 'ucb_tuned',
        'parameters': { 'average': {'type': 'keep_all'}}
    }

@enable_norm
@enable_scale
@enable_stationary_ucbe
def stationary_softmax(batch_size, params):
    return {
        'type': 'softmax',
        'parameters': { 'alpha': float(params[0]) }
    }


STATIONARY_LEARNING = {
    'ucb': stationary_ucb,
    'greedy': stationary_greedy,
    'ucb_tuned': stationary_ucb_tuned,
    'softmax': stationary_softmax
}


def stationary_learning_gen(batch_size, params):
    def helper(algo):
        return STATIONARY_LEARNING[algo](batch_size, params)
    return helper


def create_stationary_source(source):
    return path.join(current_dir(), source)


def get_kw_params(params):
    return {x[0]: x[1] 
        for x in [p.split('-') for p in params] if len(x) == 2}

def enable_average(func):
    @wraps(func)
    def helper(batch_size, params):
        average = {'type': 'keep_all'}
        kw_params = get_kw_params(params)

        if 'exp' in kw_params:
            average = {'type': 'exponential', 'gamma': float(kw_params['exp'])}
        if 'sw' in kw_params:
            average = {
                'type': 'sliding_window', 
                'num_steps': float(kw_params['sw']) / batch_size 
            }
        return func(batch_size, params, average)
    return helper


def enable_nonstationary_ucbe(func):
    @wraps(func)
    def helper(batch_size, params):
        learning = func(batch_size, params)
        kw_params = get_kw_params(params)

        if 'ucbe' in kw_params:
            return {
                'type': 'combined',
                'parameters': {
                    'exploiter': learning,
                    'explorer' : {
                        'type': 'ucb_e',
                        'parameters': {'a': 2.0 * float(kw_params['ucbe']) / batch_size }
                    }
                }
            }
        else:
            return learning
    return helper

@enable_norm
@enable_scale
@enable_nonstationary_ucbe
@enable_average
def nonstationary_ucb(batch_size, params, average):
    return {
        'type': 'ucb',
        'parameters': {
            'average': average, 
            'ksi': 2.0,
            'restricted_exploration': 'rexp' in params
        }
    }


@enable_norm
@enable_scale
@enable_nonstationary_ucbe
@enable_average
def nonstationary_ucb_tuned(batch_size, params, average):
    return {
        'type': 'ucb_tuned',
        'parameters': { 'average': average }
    }


@enable_norm
@enable_scale
@enable_nonstationary_ucbe
@enable_average
def nonstationary_greedy(batch_size, params, average):
    return {
        'type': 'epsilon_greedy',
        'parameters' : {
            'epsilon': float(params[0]),
            'average' : average,
            'delta': 0.0,
            'sigma': 1.0
        }
    }


def nonstationary_dgp(batch_size, params):
    return {
        'type': 'dgp_ucb',
        'parameters': { 'delta': float(params[0]), 'ksi': 0.0, 'average': None }
    }


@enable_average
def nonstationary_dgp_ucb(batch_size, params, average):
    return {
        'type': 'dgp_ucb',
        'parameters': { 
            'delta': float(params[0]), 
            'ksi': 2.0,
            'average': average if 'ucbcnt' in params else None 
        }
    }


NONSTATIONARY_LEARNING = {
    'ucb': nonstationary_ucb, 
    'ucb_tuned': nonstationary_ucb_tuned,
    'greedy': nonstationary_greedy,
    'dgp': nonstationary_dgp,
    'dgp_ucb': nonstationary_dgp_ucb
}

def nonstationary_learning_gen(batch_size, params):
    def helper(algo):
        return NONSTATIONARY_LEARNING[algo](batch_size, params)
    return helper

def create_nonstationary_source(batch_size, phase_size, all_pairs=False):
    if phase_size % batch_size != 0:
        print("WARN! batch_size does not divide phase_size")
    num_stat_steps = calc_num_stat_steps(batch_size)
    num_phases = FIXED_PARAMS.num_time_steps / phase_size
    if (phase_size / batch_size) % (num_stat_steps / num_phases) != 0:
        print("WARN! num_stat_steps does not divide phase_size")

    if all_pairs:
        sequence = [
           'tests/v_vs.json', 'tests/vs_vs.json', 'tests/d_vs.json',
           'tests/v_vs.json', 'tests/d_vs.json', 'tests/vs_vs.json',
           'tests/v_vs.json'
        ]
    else:
        sequence = ['tests/v_vs.json', 'tests/vs_vs.json', 'tests/d_vs.json']

    if num_phases % len(sequence) != 0:
        print("WARN! num_phases does not divide source sequence length")

    return {
        'type': 'sequence',
        'parameters': [
            { 
                'src': path.join(current_dir(), source),
                'num_steps': phase_size
            } for source in sequence
        ]
    }


def run(config):
    with NamedTemporaryFile('w') as cfg_file:
        json.dump(config, cfg_file)
        cfg_file.flush()
        return json.loads(subprocess.run([
                './build_release/src/schad',
                '--experiment-config',
                cfg_file.name], 
                stdout=subprocess.PIPE,
                stderr=None if FIXED_PARAMS.num_processes == 1 else subprocess.DEVNULL
                ).stdout)


def common_filename_opts(batch_size, num_simulated, algo):
    return [
        'nts{}'.format(FIXED_PARAMS.num_time_steps)
    ] + (['nss{}'.format(FIXED_PARAMS.num_stat_steps)] 
        if FIXED_PARAMS.num_stat_steps is not None else []
    ) + [
        'ns{}'.format(num_simulated),
        'nr{}'.format(FIXED_PARAMS.num_runs),
        'bs{}'.format(batch_size),
        'algo{}'.format(algo)
    ]


def save(filename_opts, result):
    filename = '_'.join(filename_opts) + '.json'

    if FIXED_PARAMS.directory is not None:
        filename = path.join(FIXED_PARAMS.directory, filename)
    if not path.exists(FIXED_PARAMS.directory):
        os.mkdir(FIXED_PARAMS.directory)

    print(f'Saving {filename}')
    with open(filename, 'w') as f:
        json.dump(result, f)
    return filename


def run_stationary(batch_size, num_simulated, algo, source):
    algo_name, *params = algo.split('=')
    result = run(create_pure_or_learning(
        batch_size, num_simulated, algo_name,
        create_stationary_source(source),
        stationary_learning_gen(batch_size, params)
        ))  
    filename_opts = ['stationary'] + common_filename_opts(batch_size, num_simulated, algo)
    # Hack to keep old format
    filename_opts = (filename_opts[:-1]
        + ['src{}'.format(path.splitext(path.basename(source))[0])]
        + [filename_opts[-1]]
    )
    return save(filename_opts, result)


def run_nonstationary(phase_size, batch_size, num_simulated, algo, all_pairs):
    algo_name, *params = algo.split('=')
    result = run(create_pure_or_learning(
        batch_size, num_simulated, algo_name,
        create_nonstationary_source(batch_size, phase_size, all_pairs),
        nonstationary_learning_gen(batch_size, params)
        ))  
    filename_opts = (['nonstationary', f'ps{phase_size}'] 
            + (['allpairs'] if all_pairs else [])
            + common_filename_opts(batch_size, num_simulated, algo))
    return save(filename_opts, result)


def run_n_save_all(tasks, save):
    try:
        paths = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks)) 
        if save is not None:
            with tarfile.open(fileobj=save, mode='w:gz') as tar:
                for path in paths:
                    tar.add(path)

    finally:
        FIXED_PARAMS.executor.shutdown()


@hello.command()
@click.option('--algo', type=str, required=False, multiple=True)
@click.option('--source', type=str, required=True)
@click.option('--num-simulated', type=int, required=True, multiple=True)
@click.option('--batch-size', type=int, required=True, multiple=True)
@click.option('--save', type=click.File('wb'), required=False)
def stationary(algo, source, num_simulated, batch_size, save):
    loop = asyncio.get_event_loop()

    result_tasks = []
    for ns, bs, a in product(num_simulated, batch_size, algo):
        result_tasks.append(loop.run_in_executor(
            FIXED_PARAMS.executor, run_stationary, bs, ns, a, source))
    run_n_save_all(result_tasks, save)


@hello.command()
@click.option('--algo', type=str, required=False, multiple=True)
@click.option('--num-simulated', type=int, required=True, multiple=True)
@click.option('--batch-size', type=int, required=True, multiple=True)
@click.option('--phase-size', type=int, required=True, multiple=True)
@click.option('--all-pairs', is_flag=True)
@click.option('--save', type=click.File('wb'), required=False)
def nonstationary(algo, num_simulated, batch_size, phase_size, save, all_pairs):
    loop = asyncio.get_event_loop()

    result_tasks = []
    for ps, ns, bs, a in product(phase_size, num_simulated, batch_size, algo):
        result_tasks.append(loop.run_in_executor(
            FIXED_PARAMS.executor, run_nonstationary, ps, bs, ns, a, all_pairs))
    run_n_save_all(result_tasks, save)


if __name__ == '__main__':
    hello()

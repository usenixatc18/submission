# Towards declarative self-adapting buffer management (code for evaluations)

## Requirements

For compiling simulation program:

 * Recent C++ compiler (e.g., [GCC][gcc] 7 or higher)
 * [CMake][cmake]
 * [Boost C++ Libraries][boost]
 * [nlohmann's JSON][json]

For the python scripts: 

 * Recent [python][python] version (3.6 and higher)
 * [Click][click] library
 * [NumPy][numpy]
 * [Matplotlib][matplotlib]


## Compilation

 1. Clone this repository and change to the directory:

    ```bash
    git clone https://github.com/usenixatc18/submission.git simulator
    cd simulator
    ```

 2. Compile

    ```bash
    mkdir build_release
    cd build_release
    cmake ..
    make
    cd ..
    ```

## Generate data for stationary plots 2(a-h)

Setup algorithms for stationary case:

```bash
ST_ALGOS=(--algo ucb_tuned=norm --algo ucb=norm --algo v --algo vs --algo d)
COMMON=(--num-time-steps 1000000 --num-stat-steps 200 --num-runs 100)
```

### Figures 2(a-d)

Traffic preferring PQ(v):

```bash
./run.py ${COMMON[*]} stationary --source tests/v_vs.json \
    ${ST_ALGOS[*]} --batch-size 1 \
    --num-simulated 0 --num-simulated 1 --num-simulated 2 
```

Traffic preferring PQ(v/s):

```bash
./run.py ${COMMON[*]} stationary --source tests/vs_vs.json \
    ${ST_ALGOS[*]} --batch-size 1 \
    --num-simulated 0 --num-simulated 1 --num-simulated 2 
```

Traffic preferring PQ(-d):

```bash
./run.py ${COMMON[*]} stationary --source tests/d_vs.json \
    ${ST_ALGOS[*]} --batch-size 1 \
    --num-simulated 0 --num-simulated 1 --num-simulated 2 
```

### Figures 2(e-h)

Traffic preferring PQ(v):

```bash
./run.py ${COMMON[*]} stationary --source tests/v_vs.json \
    ${ST_ALGOS[*]} --num-simulated 0 \
    --batch-size 1 --batch-size 2 --batch-size 4 --batch-size 8 \
    --batch-size 16 --batch-size 32
```

Traffic preferring PQ(v/s):

```bash
./run.py ${COMMON[*]} stationary --source tests/vs_vs.json \
    ${ST_ALGOS[*]} --num-simulated 0 \
    --batch-size 1 --batch-size 2 --batch-size 4 --batch-size 8 \
    --batch-size 16 --batch-size 32
```

Traffic preferring PQ(-d):

```bash
./run.py ${COMMON[*]} stationary --source tests/d_vs.json \
    ${ST_ALGOS[*]} --num-simulated 0 \
    --batch-size 1 --batch-size 2 --batch-size 4 --batch-size 8 \
    --batch-size 16 --batch-size 32
```

### Figures 2(h-f)

Traffic preferring PQ(v):

```bash
./run.py ${COMMON[*]} --max-reward 100 stationary --source tests/v_vs.json \
    --num-simulated 0 --batch-size 1 --algo v \
    --algo ucb_tuned --algo ucb_tuned=norm --algo ucb_tuned=scale \
    --algo ucb --algo ucb=norm --algo ucb=scale
```

Traffic preferring PQ(v/s):

```bash
./run.py ${COMMON[*]} --max-reward 256 stationary --source tests/vs_vs.json \
    --num-simulated 0 --batch-size 1 --algo vs \
    --algo ucb_tuned --algo ucb_tuned=norm --algo ucb_tuned=scale \
    --algo ucb --algo ucb=norm --algo ucb=scale
```

Traffic preferring PQ(v/s):

```bash
./run.py ${COMMON[*]} --max-reward 685 stationary --source tests/d_vs.json \
    --num-simulated 0 --batch-size 1 --algo d \
    --algo ucb_tuned --algo ucb_tuned=norm --algo ucb_tuned=scale \
    --algo ucb --algo ucb=norm --algo ucb=scale
```

## Generate data for non-stationary plots 2(l-p)

Set up common parameters:

```bash
COMMON=(--num-time-steps 30000000 --num-stat-steps 300 --num-runs 10)
NONSTAT=(--phase-size 10000000 --batch-size 5 --algo v --algo vs --algo d)
```

### Figures 2(l-n)

```bash
./run.py ${COMMON[*]} nonstationary ${NONSTAT[*]} \
    --num-simulated 0 --num-simulated 1 --num-simulated 2 \
    --algo ucb_tuned=norm=exp-0.99999 --algo dgp_ucb=0.9999
```

### Figure 2(o)

```bash
./run ${COMMON[*]} nonstationary ${NONSTAT[*]} --num-simulated 0 \
    --algo ucb_tuned=norm=exp-0.99 --algo ucb_tuned=norm=exp-0.999 \
    --algo ucb_tuned=norm=exp-0.9999 --algo ucb_tuned=norm=exp-0.99999 \
    --algo ucb_tuned=norm=exp-0.99999
```

### Figure 2(p)

```bash
./run ${COMMON[*]} nonstationary ${NONSTAT[*]} --num-simulated 0 \
    --algo dgp_ucb=exp-0.99 --algo dgp_ucb=exp-0.999 \
    --algo dgp_ucb=exp-0.9999 --algo dgp_ucb=exp-0.99999 \
    --algo dgp_ucb=exp-0.99999
```

**Note** for performance pass before all other parameters `--num-processes k`,
where `k` is a desired level of parallelism.

### Using the results

The results will appear in `./data` directory in a form of `.json` files

To plot the graphs with *cumulative reward* use (as in Fig 2(a-k)) use:

```bash
./plot plot --combined --plot-type relative_cum_reward [JSONs]
```

To plot the graphs with *reward* use (as in Fig 2(l-p)) use:

```bash
./plot plot --combined --plot-type relative_reward [JSONs]
```

Here `[JSONs]` is a listed of `.json` files (e.g., generate by commands shown above)

**Note** optimal algorithms (or all three in case of non-stationary) 


### Final remarks

 * All scripts support `--help` option for additional options.
 * Every `.json` result has all the information about parameters used for 
   producing it in the `experiment` entry.

[gcc]: https://gcc.gnu.org/
[click]: http://click.pocoo.org/
[cmake]: https://cmake.org/
[boost]: http://www.boost.org/
[json]: https://github.com/nlohmann/json
[python]: https://www.python.org/
[numpy]: http://www.numpy.org/
[matplotlib]: https://matplotlib.org/




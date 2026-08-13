# WeakTTC

WeakTTC experiments for reassigning AgentE participants to AgentC resources under
weak preferences and capacity constraints.

## Folders

- `src_new/`: current implementation and runnable entry point.
- `src_new/test/`: small debug/test scripts.
- `data/`: input data files.
- `results/`: generated experiment outputs and plots.
- `tools/`: plotting utilities.

## Install Libraries

```bash
pip install -r requirements.txt
python src_new/main.py
```

## Run

From the repository root:

```bash
python src_new/main.py
```

`src_new/main.py` initializes an example instance, runs `WeakTTC`, `TTC`, and
`ReactTTC_variant`, then prints total rank improvement and runtime for each.

## Experiment Runners

Synthetic-data runners:

```bash
python src_new/run_varying_agentsize.py
python src_new/run_varying_agentsize_fixed_resource.py
python src_new/run_varying_resource_ratio.py
python src_new/run_varying_capacity.py
python src_new/run_varying_max_rank_size.py
```

Real EV charging-point runner:

```bash
python src_new/run_real_ev_charging.py
```

The real runner reads `data/combined_data_jd200_1.csv`, ignores `type=d`,
treats `type=c` as EVs, treats `type=f` as charging points, and builds
preferences from real EV-to-charging-point distances in
`data/distance_matrix_jd200_1.csv`.

It runs the real-data versions of the synthetic experiments in one script:

- varying total EV with fixed EV-to-charging-point ratio
- varying total charging points with fixed EV count
- varying charging-point capacity
- varying profile size, i.e. charging points per preference class

By default, the real runner uses a random feasible initial assignment. To use nearest-feasible assignment:


Run one real experiment family instead of all four:

```bash
python src_new/run_real_ev_charging.py --experiment varying_ev
python src_new/run_real_ev_charging.py --experiment varying_cp
python src_new/run_real_ev_charging.py --experiment varying_capacity
python src_new/run_real_ev_charging.py --experiment varying_profile_size
```

## Plotting

Individual plotters live in `tools/`:

```bash
python tools/plot_varying_agentsize.py
python tools/plot_varying_agentsize_fixed_resource.py
python tools/plot_varying_resource_ratio.py
python tools/plot_varying_capacity.py
python tools/plot_varying_max_rank_size.py
python tools/plot_real_ev_charging.py
```

To regenerate every available plot from existing CSV files in `results/`
without rerunning any experiment:

```bash
python tools/plot_all_results.py
```

Useful options:

```bash
python tools/plot_all_results.py --format png
python tools/plot_all_results.py --results-dir /path/to/results
```

## Documentation

```bash
pip install pdoc
```

Generate docs:

```bash
PYTHONPATH=src_new:tools pdoc src_new/*.py tools/*.py -o docs
```

Then open `docs/index.html` in a browser. This documents the algorithm modules,
runners, and plotters without requiring a separate configuration file.


## Virtual Environment

If using the included local virtual environment:

```bash
source env_EV_TTC/bin/activate
```
If you are using the included virtual environment and `pdoc` is installed there:

```bash
PYTHONPATH=src_new:tools env_EV_TTC/bin/pdoc src_new/*.py tools/*.py -o docs
```

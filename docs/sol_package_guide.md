# Installing Python Packages on Sol — A Personal Reference Guide

> Source material: ASU Research Computing Confluence pages on Mamba environment management, Python package installation methods, common issues, SBATCH scripts, interactive shells, and job arrays. This guide condenses those into one document with the Sol-specific gotchas called out explicitly, so you don't have to relearn them at 2 AM.

---

## 1. The five rules you must never break

These five rules cover ~95% of the failure modes the documentation warns about. Internalize them before doing anything else.

1. **Never install packages on a login node.** Sol's login nodes are shared and not for compute. Every `mamba create`, `mamba install`, or `pip install` must run inside an `interactive` session (or an sbatch job).
2. **Never use `conda activate` or `mamba activate`. Always use `source activate <env>`.** Sol's `mamba/latest` module is a custom wrapper. The `conda activate` and `mamba activate` paths inject shell hooks into `~/.bashrc` that break the supercomputing environment. The only blessed activation pattern on Sol is `module load mamba/latest && source activate <env>`.
3. **Never use `pip install --user`.** It dumps packages into `~/.local/lib/pythonX.Y/site-packages`, which is shared across *every* environment with a matching Python version and silently corrupts all of them. If you've ever done this by accident, the fix is in §7.
4. **Never use the `defaults` conda channel.** Use `-c conda-forge` (and add a second `-c` for specialized channels like `bioconda` or `nvidia` when needed). The Sol Mamba docs call this out explicitly, citing the upstream Mamba troubleshooting guide.
5. **Always put `#SBATCH --export=NONE` in every sbatch script.** Without it, sbatch inherits a half-configured Mamba state from the shell that submitted the job. The symptom is always the same: `source activate <env>` silently no-ops inside the job, `which python` shows `/packages/apps/mamba/.../bin/python` instead of your env's Python, and your imports fail with a confusing traceback from `/etc/python/sitecustomize.py`. This was the bug we spent hours debugging. Don't do it again.

---

## 2. Sol's environment manager: Mamba (not conda, not pip)

Sol uses **Mamba** as its package manager — a parallel C++ reimplementation of conda that's much faster at dependency resolution. Anywhere you'd type `conda`, type `mamba` instead. The Research Computing team explicitly discourages `pip` except where unavoidable, because pip doesn't respect the safeguards that make a multi-user system stable.

### Loading Mamba
Always start with:
```bash
module load mamba/latest
```
This makes the `mamba` and `source activate` commands available. It does *not* activate any environment by itself.

### Listing existing environments
```bash
mamba info --envs
```
Public, admin-maintained envs live under `/packages/envs` and are read-only. Your personal envs live under `~/.conda/envs` by default. Group-shared envs can live anywhere on `/data` and are activated by full path: `source activate /data/yourgroup/.conda/envs/somename`.

### The base environment is off-limits
If your prompt ever shows `(base)`, deactivate it before doing anything:
```bash
source deactivate
```
Never install into `base`. It's not yours.

---

## 3. Creating environments — the right way

### Where to do it
Get an interactive session first. For pure-CPU package installs, `htc` is fastest to schedule:
```bash
interactive -t 60 -p htc
module load mamba/latest
```
If the env will use a GPU and you want to verify CUDA works in the same session, request a GPU instead:
```bash
interactive -p general -q public -G a100:1 -c 8 --mem=32G -t 0-02:00
module load mamba/latest
```
(The `general` partition auto-remaps to `public/public` right now and will be removed entirely after the next maintenance — `-p public -q public` is the future-proof form.)

### The single-command rule
**Install everything in one `mamba create` call.** The Sol docs are emphatic about this: it maximizes stability and minimizes build time because Mamba can resolve all dependencies at once instead of layering installs and constantly rewriting the env. Do this:

```bash
mamba create -y -n myenv -c conda-forge python=3.11 numpy scipy matplotlib pandas
```

Don't do this:
```bash
mamba create -n myenv python=3.11
source activate myenv
mamba install numpy
mamba install scipy
mamba install matplotlib
# ← every step risks dependency conflicts and slows down resolution
```

### Channels
- `-c conda-forge` is your default. Always include it.
- Add other channels with additional `-c` flags: `-c conda-forge -c bioconda multiqc`.
- **Never** use the `defaults` channel.
- To find the right channel for a package, search it on [anaconda.org](https://anaconda.org). The page will tell you which channel hosts it.

### Creating in a non-default location
If you want the env to live under `/data` (e.g., for a shared lab env) instead of `~/.conda/envs`, use `-p` with a full path instead of `-n` with a name:
```bash
mamba create -p /data/yourgroup/envs/sharedenv -c conda-forge python=3.11 ...
```
Be careful — envs in non-default locations are easier to lose track of. Always double-check the `Prefix:` line Mamba prints before confirming the install.

### Activating and using
```bash
source activate myenv          # name-based
source activate /data/yourgroup/envs/sharedenv   # path-based
```
Your prompt will show `(myenv)` on the left when an env is active. To check what's installed:
```bash
mamba list
```
To deactivate:
```bash
source deactivate
```

### Adding packages to an existing private env
```bash
module load mamba/latest
source activate myenv
mamba install -c conda-forge new_package
```
Still inside an interactive session. Still single-command if you're adding multiple.

### Cloning a public env so you can modify it
Public envs are read-only. To extend one, clone it via YAML export:
```bash
module load mamba/latest
source activate publicEnvName
mamba env export --from-history --no-builds -n publicEnvName > ~/myenv.yaml
source deactivate
mamba env create -n myownenv --file ~/myenv.yaml
```
Drop `--from-history --no-builds` if you want exact version pins (warning: pins from old envs frequently cause conflicts).

### Removing an env
```bash
mamba remove -n myenv --all
mamba clean --all
```

---

## 4. When you actually need pip

Some packages (notably modern PyTorch, TensorFlow, JAX) are only distributed via PyPI wheels, or their official install instructions point at pip. The rule is simple:

**`pip install` is only acceptable inside an activated mamba env. Period.**

```bash
interactive -t 60 -p htc
module load mamba/latest
mamba create -n myenv -c conda-forge python=3.12     # create with mamba first
source activate myenv                                 # activate
pip install some-pypi-only-package                    # then pip — and only then
```

When pip runs inside an activated env, it installs into `~/.conda/envs/myenv/lib/pythonX.Y/site-packages`, which is properly scoped. Outside an activated env, pip installs into `~/.local/lib`, which is the source of every "Jupyter is in a bad state" / "imports work in interactive but fail in sbatch" bug report on Sol.

**Never use `pip install --user`** — even inside an activated env, `--user` overrides the env's site-packages and dumps to `~/.local/lib`. There is no situation on Sol where `--user` is correct.

### CUDA-aware pip installs
If you're installing a GPU package via pip (e.g., `pip install torch ... --index-url https://download.pytorch.org/whl/cu126` or `pip install tensorflow[and-cuda]`), load a CUDA module *before* pip-installing so the package can find the toolkit:
```bash
interactive -t 60 -p htc
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
mamba create -n myenv -c conda-forge python=3.12
source activate myenv
pip install tensorflow[and-cuda]
```
For Numba CUDA the recommended path is different — install `cudatoolkit` from conda-forge as part of the env itself (`mamba create ... cudatoolkit=11.8`). No `module load cuda` needed at runtime; the env carries its own toolkit.

---

## 5. SBATCH scripts: the canonical Sol pattern

Every sbatch script you write should follow this skeleton. Deviating from it is how you generate the bug we spent a day debugging.

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH -p public                  # partition (public is the future-proof form)
#SBATCH -q public                  # QOS
#SBATCH -N 1                       # nodes
#SBATCH -c 8                       # cores per task
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00              # d-hh:mm:ss
#SBATCH -o logs/myjob_%j.out       # %j = job id
#SBATCH -e logs/myjob_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jfigger@asu.edu
#SBATCH --export=NONE              # ← MANDATORY. Do not omit.

module load mamba/latest
source activate myenv

# Sanity check — uncomment for first run, remove once verified
# which python    # should show /home/jfigger/.conda/envs/myenv/bin/python

cd "${SLURM_SUBMIT_DIR}"
python myscript.py
```

### Things that look like good practice but break sbatch on Sol

- **`set -eo pipefail` (and especially `set -u`)**: conda/mamba activation scripts touch unset variables internally. With `set -u` enabled, activation aborts halfway through and you get a Python that's almost-but-not-quite the env's Python. Remove these from your scripts.
- **`conda activate` / `mamba activate`**: same as in interactive sessions — they don't work and they pollute your shell config. Use `source activate`.
- **Omitting `--export=NONE`**: see Rule 5 in §1. The symptoms are silent and confusing. Always include it.
- **`#SBATCH -p general`**: the `general` partition is being deprecated. It still auto-remaps to `public/public` for now but will hard-fail after the next maintenance window. Use `-p public -q public`.

### GPU jobs
Add `-G 1` for one GPU, or `-G a100:1` if you need a specific model. The chat node will report the GPU via `nvidia-smi` once allocated.

### Job arrays
Add `#SBATCH --array=0-31` (or 1-32, however you want to index). Inside the script, `$SLURM_ARRAY_TASK_ID` holds the current index. Output files use `%A_%a` for `<arrayJobID>_<taskID>`:
```bash
#SBATCH --array=0-31
#SBATCH -o logs/myjob_%A_%a.out
#SBATCH -e logs/myjob_%A_%a.err
```
Best practice for arrays: build a manifest file listing one input per line, then index into it with `$SLURM_ARRAY_TASK_ID`. Or, for parameter sweeps generated in code (like the MADI shard scheme), pass `$SLURM_ARRAY_TASK_ID` directly to your script as a CLI arg.

### Submitting and managing
```bash
sbatch myjob.sbatch                          # submit
sbatch --array=0 myjob.sbatch                # submit array but only task 0 (testing)
sbatch -t 12:00:00 myjob.sbatch              # override walltime at submit time
squeue -u $USER                              # what's queued/running
myjobs                                       # nicer formatted version
thisjob <jobID>                              # estimated start time, full info
seff <jobID>                                 # CPU/mem efficiency for COMPLETED jobs
scancel <jobID>                              # cancel
scontrol update job <jobID> ReqCores=4       # modify a PENDING job
```

### Other useful Sol-specific commands
- `myquota` — your scratch quota
- `myfairshare` — your fair-share priority score
- `mkjupy <envname>` — turn a mamba env into a Jupyter kernel
- `ns` — text-mode cluster status
- `sinfo` — partition/node availability
- `remove_conda_from_bashrc` — undoes accidental `conda init` damage

---

## 6. Interactive sessions

The `interactive` command is a wrapper around `salloc` that drops you into a shell on a compute node. Defaults: `htc` partition, `public` QOS, 1 core, 4-hour walltime.

```bash
interactive                                   # defaults
interactive -c 8 -t 0-4:00                    # 8 cores, 4 hours
interactive -G 1                              # 1 GPU
interactive -p public -q public -G a100:1 -c 8 --mem=32G -t 0-02:00   # full spec
```

You know you're on a compute node when your prompt switches from `login01` (or similar) to a node name like `c001` or `cg001`. Don't run `mamba install` until you see that change.

X11 forwarding works automatically inside interactive sessions if you SSH'd in with `ssh -X`. For anything graphical, the Sol web portal is more reliable than X11.

---

## 7. Recovery: fixing a broken Python state

If your env starts behaving strangely — imports failing, version mismatches, "Bad State" in Jupyter, sbatch jobs that work in one place and fail in another — the cause is almost always one of:

### A. `~/.local/lib` is contaminated
Check it:
```bash
cd ~/.local/lib
ls
```
If you see any `python3.X` directories, that's leftover from `pip install --user` or pip-without-an-env. Nuke or rename them:
```bash
mv python3.11 python3.11.bak     # safer (rename)
# OR
rm -rf python3*                   # nuke (irreversible)
mamba clean --all
```
Then reinstall whatever you needed *into your mamba env* with `source activate myenv && mamba install ...`.

### B. `~/.bashrc` has conda init code in it
If you ever ran `conda init` (or it ran itself), your `~/.bashrc` now has a conda hook block that fights with Sol's mamba module. Fix:
```bash
remove_conda_from_bashrc
source ~/.bashrc
```

### C. The env builds fine but sbatch can't activate it
99% of the time this is a missing `#SBATCH --export=NONE`. Add it, resubmit. If `which python` in the script's output still doesn't show your env's Python after that, *then* go looking deeper.

### D. The env itself is broken beyond repair
Don't fight it — rebuild:
```bash
interactive -t 60 -p htc
module load mamba/latest
mamba remove -n myenv --all
mamba clean --all
mamba create -y -n myenv -c conda-forge python=3.X pkg1 pkg2 ...   # single command
```

---

## 8. Quick reference: the right command for each situation

| You want to... | Do this |
|---|---|
| Get to a compute node | `interactive` (add `-G 1` for GPU, `-p public -q public` to be explicit) |
| Load mamba | `module load mamba/latest` |
| List envs | `mamba info --envs` |
| Create env | `mamba create -y -n NAME -c conda-forge python=X.Y pkg1 pkg2 ...` (single command, in interactive) |
| Activate env | `source activate NAME` (never `conda activate`, never `mamba activate`) |
| Deactivate | `source deactivate` |
| Add a package | `source activate NAME && mamba install -c conda-forge pkg` (in interactive) |
| Pip install (only when necessary) | `source activate NAME && pip install pkg` (NEVER `--user`) |
| Remove env | `mamba remove -n NAME --all && mamba clean --all` |
| Submit a job | `sbatch script.sbatch` |
| Submit one array task only (test) | `sbatch --array=0 script.sbatch` |
| Watch the queue | `squeue -u $USER` or `myjobs` |
| Cancel | `scancel <jobID>` |
| Post-mortem efficiency | `seff <jobID>` |
| Tail a running job's stdout | `tail -f logs/jobname_*_0.out` |

---

# Appendix: MADI environment setup, login → sbatch

Below are the exact commands to bring the MADI environment up from scratch and submit the dense library build. Run them in order. Lines starting with `#` are comments — don't paste those.

## Step 0: From your laptop

```bash
ssh jfigger@sol.asu.edu
```

## Step 1: Hygiene check (one-time, skip on subsequent setups)

Make sure no stale `~/.local/lib` packages will interfere with the env you're about to build.

```bash
ls ~/.local/lib 2>/dev/null
# If you see any python3.X directories from previous mishaps, rename them:
# mv ~/.local/lib/python3.11 ~/.local/lib/python3.11.bak
```

## Step 2: Get an interactive GPU session for the build

We want a GPU on this session because we're going to verify that Numba can see CUDA before submitting the array. The `general` partition still auto-remaps but `public` is future-proof.

```bash
interactive -p public -q public -G a100:1 -c 8 --mem=32G -t 0-02:00
```

Wait until your prompt switches from `login0X` to a compute node name (e.g., `cg012`). If it doesn't allocate within a minute or two, you can check the queue with `squeue -u $USER` from another shell, or downgrade to `interactive -p htc -c 8 -t 0-01:00` (no GPU — you just won't be able to run the CUDA verification step on the same node).

## Step 3: Build the `madi` environment in a single mamba call

```bash
module load mamba/latest

mamba create -y -n madi -c conda-forge \
    python=3.11 \
    numpy=1.26 \
    scipy \
    matplotlib \
    nibabel \
    numba \
    cudatoolkit=11.8 \
    tqdm

source activate madi
```

Single-command creation is intentional — it lets Mamba resolve the whole dependency graph in one pass and produces a far more stable env than chained `mamba install` calls.

## Step 4: Verify the env

The first line of output must be `/home/jfigger/.conda/envs/madi/bin/python`. If it's anything else (especially `/packages/apps/mamba/...`), the activation didn't take and you should not proceed — drop back to §1 Rule 2 and figure out why.

```bash
which python

python -c "
import numpy, scipy, matplotlib, nibabel, numba
from numba import cuda
print('imports OK')
print('CUDA available:', cuda.is_available())
print('GPU:', cuda.get_current_device().name.decode() if cuda.is_available() else 'NONE')
"
```

Expected output:
```
/home/jfigger/.conda/envs/madi/bin/python
imports OK
CUDA available: True
GPU: NVIDIA A100-SXM4-80GB
```

If you see that, the env is good. Release the interactive session — it's not needed for submission:

```bash
exit
```

You're now back on a login node.

## Step 5: Navigate to the repo and prepare directories

```bash
cd /scratch/jfigger/madi/custom_madi
mkdir -p logs libraries
```

## Step 6: Verify the sbatch script is correct

Quickly inspect the script to confirm `--export=NONE` is present and partition is `public`:

```bash
grep -E "^#SBATCH (--export|-p|-q)" build_lib.sbatch
```

You should see:
```
#SBATCH -p public
#SBATCH -q public
#SBATCH --export=NONE
```

If `--export=NONE` is missing, **fix it before submitting**.

## Step 7: Test submission — one array task only

Submit just shard 0 first. This validates the full environment activation path inside sbatch, not just inside interactive (these are different and fail in different ways).

```bash
sbatch --array=0 build_lib.sbatch
squeue -u $USER
```

Note the job ID. Once it transitions from `PENDING` to `RUNNING`, tail its output:

```bash
tail -f logs/madi_dense_*_0.out
```

The first three lines you should see are:

```
/home/jfigger/.conda/envs/madi/bin/python
imports OK
CUDA: True
GPU: NVIDIA A100-SXM4-80GB
```

If you see those, `Ctrl-C` the tail. The activation works; sharding is wired correctly. You can either let shard 0 finish (it will get checkpointed and merged later) or cancel it with `scancel <jobID>` to free the slot before submitting the full array.

If you see `/packages/apps/mamba/.../bin/python` on the first line, **stop and debug**. Check that `--export=NONE` is in the script, that there's no `set -u` or `set -eo pipefail`, and that the script doesn't try to run `conda activate` or `mamba activate` anywhere.

## Step 8: Submit the full array

```bash
sbatch build_lib.sbatch
squeue -u $USER
```

This launches all 32 shards (`--array=0-31`). With ensemble reuse and the per-(ρ,V)-group checkpointing, each shard will build its assigned groups and write `libraries/madi_dense.shard000.npz` … `libraries/madi_dense.shard031.npz`.

Monitor with:
```bash
squeue -u $USER
tail -f logs/madi_dense_*_0.out      # any specific task
ls -lh libraries/                     # watch shards land
```

## Step 9: Post-mortem and merge

When everything finishes:
```bash
seff <jobID>                          # check CPU/mem efficiency of a representative task
ls libraries/madi_dense.shard*.npz | wc -l    # should be 32
python merge_shards.py libraries/madi_dense.shard*.npz -o libraries/madi_dense.npz
```

If any shard hit the walltime, re-shard at higher granularity (`--array=0-63`, change `--n-shards 64` in the python call) and resubmit — the per-group checkpoints in `library.py` make this safe and cheap.

---

That's it. The guide above is meant to live in the repo (or your notes) as the canonical reference for any future Sol work, not just MADI.

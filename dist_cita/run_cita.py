#!/usr/bin/env python
"""
Execute run_cita.ipynb and keep its outputs: `python run_cita.py` (in the activated venv, from the bundle directory).

    python run_cita.py                 the default run (every cell, per-cell time budgets; ~2.5 h bound)
    CITA_QUICK=1 python run_cita.py    the ~20-minute first pass (same as --quick)
    python run_cita.py --inproc        run the cells in this process instead of an ipykernel (the fallback when
                                       nbclient / ipykernel cannot start; outputs are still saved into the notebook)

Writes run_cita_executed.ipynb (saved after every cell, so a crash keeps the cells that ran), run_cita.log (this
script's own log) and, after the final save, refreshes the results tarball the notebook's last cell made so that it
contains the complete executed notebook. The notebook's knobs (CITA_QUICK, CITA_MAX_N, CITA_BUDGET_SCALE, CITA_FULL,
CITA_SERIES, CITA_OUT, CITA_LABEL, CITA_SMOKE) are environment variables; see README.md.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import tarfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def run_nbclient(nb_path, out_path):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": HERE}})

    def start(**kw):
        print(f"[run_cita] {time.strftime('%H:%M:%S')} cell {kw.get('cell_index')} started (watch results/.../logs/*.log for the rows)", flush=True)

    def save(**kw):
        try:
            nbformat.write(nb, out_path)
            idx = kw.get("cell_index")
            cell = kw.get("cell")
            if cell is not None:                                                # echo the cell's text output to the terminal / run_cita.log
                for o in cell.get("outputs", []):
                    if o.get("output_type") == "stream":
                        print(o.get("text", ""), end="", flush=True)
                    elif o.get("output_type") == "error":
                        print("\n".join(o.get("traceback", [])), flush=True)
            print(f"[run_cita] {time.strftime('%H:%M:%S')} saved {os.path.basename(out_path)} after cell {idx}", flush=True)
        except Exception:                                                       # noqa: BLE001
            traceback.print_exc()
    client.on_cell_start = start
    client.on_cell_executed = save
    try:
        client.execute()
    finally:
        nbformat.write(nb, out_path)


def run_inproc(nb_path, out_path):
    """Execute the code cells in this process: stdout/stderr captured as stream outputs, show_md / show_png display
    outputs collected through the notebook's `_cita_outputs` hook; the notebook is saved after every cell."""
    nb = json.load(open(nb_path))
    ns = {"__name__": "__main__", "__builtins__": __builtins__}
    count = 0
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        buf = io.StringIO()
        outputs: list = []
        ns["_cita_outputs"] = outputs
        count += 1
        outer_out, outer_err = sys.stdout, sys.stderr
        with contextlib.redirect_stdout(_Tee(outer_out, buf)), contextlib.redirect_stderr(_Tee(outer_err, buf)):
            try:
                exec(compile(src, f"<cell {i}>", "exec"), ns)                   # noqa: S102  (the bundle's own notebook)
            except BaseException:                                               # noqa: BLE001  (a cell never stops the next)
                traceback.print_exc(file=sys.stdout)
        text = buf.getvalue()
        cell["outputs"] = ([{"output_type": "stream", "name": "stdout", "text": text}] if text else []) + outputs
        cell["execution_count"] = count
        with open(out_path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"[run_cita] saved {os.path.basename(out_path)} after cell {i}", flush=True)


def refresh_tarball(out_dir_hint=None):
    """Recreate the tarball the notebook's last cell made (same name, read from <results>/last_tarball.txt), now with
    the complete executed notebook inside."""
    candidates = [out_dir_hint] if out_dir_hint else []
    candidates += [os.environ.get("CITA_OUT") or "", os.path.join(HERE, "results", "v3_cita"), os.path.join(HERE, "results", "v3_cita_smoke")]
    for out in candidates:
        if out and os.path.exists(os.path.join(out, "last_tarball.txt")):
            name = open(os.path.join(out, "last_tarball.txt")).read().strip()
            break
    else:
        print("[run_cita] no last_tarball.txt found: the notebook's last cell did not run; no tarball refreshed")
        return None
    path = os.path.join(HERE, name)
    extras = ["run_cita_executed.ipynb", "setup.log", "build_classic.log", "build_native.log", "doctor.txt", "tiers.txt",
              "tier_a.log", "pip_wheels.log", "pip_src.log", "GIT_COMMIT", "run_cita.log"]
    with tarfile.open(path, "w:gz") as tf:
        tf.add(out, arcname=os.path.relpath(out, HERE) if out.startswith(HERE) else os.path.basename(out))
        for e in extras:
            p = os.path.join(HERE, e)
            if os.path.exists(p):
                tf.add(p, arcname=e)
    print(f"[run_cita] results tarball (bring this back): {path}  ({os.path.getsize(path) / 2**20:.1f} MiB)", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--notebook", default=os.path.join(HERE, "run_cita.ipynb"))
    ap.add_argument("--output", default=os.path.join(HERE, "run_cita_executed.ipynb"))
    ap.add_argument("--quick", action="store_true", help="CITA_QUICK=1: the ~20-minute subset")
    ap.add_argument("--inproc", action="store_true", help="execute the cells in this process (no ipykernel)")
    a = ap.parse_args()
    if a.quick:
        os.environ["CITA_QUICK"] = "1"
    os.chdir(HERE)
    # the kernel's connection files go next to the bundle (a read-only or absent $HOME must not stop the run)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", os.path.join(HERE, ".jupyter_runtime"))
    os.makedirs(os.environ["JUPYTER_RUNTIME_DIR"], exist_ok=True)
    log = open(os.path.join(HERE, "run_cita.log"), "a")
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    t0 = time.time()
    print(f"[run_cita] {time.strftime('%Y-%m-%d %H:%M:%S')} notebook {a.notebook} -> {a.output}; python {sys.executable}; "
          f"CITA_QUICK={os.environ.get('CITA_QUICK', '')} CITA_SMOKE={os.environ.get('CITA_SMOKE', '')} CITA_MAX_N={os.environ.get('CITA_MAX_N', '')} "
          f"CITA_BUDGET_SCALE={os.environ.get('CITA_BUDGET_SCALE', '')} CITA_OUT={os.environ.get('CITA_OUT', '')}", flush=True)
    mode = "inproc" if a.inproc else "nbclient"
    if mode == "nbclient":
        try:
            run_nbclient(a.notebook, a.output)
        except Exception:                                                       # noqa: BLE001
            traceback.print_exc()
            print("[run_cita] nbclient / ipykernel failed (see above); falling back to in-process execution", flush=True)
            mode = "inproc"
    if mode == "inproc":
        run_inproc(a.notebook, a.output)
    print(f"[run_cita] executed notebook: {a.output}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    refresh_tarball()


if __name__ == "__main__":
    main()

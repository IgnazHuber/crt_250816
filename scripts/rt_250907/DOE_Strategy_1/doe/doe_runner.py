import os
import sys
import json
import subprocess
import tempfile
import hashlib
import signal
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Support running as a module or as a script
try:
    from .doe_spec import build_specs, build_specs_from_config
    from .metrics import compute_metrics_from_csv
except ImportError:
    # Fallback when executed directly (no package context)
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from doe_spec import build_specs, build_specs_from_config
    from metrics import compute_metrics_from_csv


def run_backtest(main_script_path: str, folder: str, spec: dict, out_dir: str,
                 stop_evt: threading.Event | None = None, hard_stop: bool = True) -> tuple[str, dict]:
    os.makedirs(out_dir, exist_ok=True)
    # Create a stable id for the spec
    spec_str = json.dumps(spec, sort_keys=True)
    spec_id = hashlib.md5(spec_str.encode('utf-8')).hexdigest()[:10]
    out_csv = os.path.join(out_dir, f"backtest_{spec['ema_variant']}_{spec['div_family']}_{spec_id}.csv")

    cmd = [
        sys.executable, main_script_path,
        '--folder', folder,
        '--doe-module', 'DOE_Strategy_1.doe.doe_strategy',
        '--doe-spec', spec_str,
        '--output', out_csv,
    ]
    # Run in workspace root so package import works; inject PYTHONPATH
    repo_root = os.path.dirname(os.path.dirname(main_script_path))
    env = os.environ.copy()
    env['PYTHONPATH'] = repo_root + (os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env else '')
    creationflags = 0
    if os.name == 'nt':
        creationflags |= getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    proc = subprocess.Popen(cmd, cwd=repo_root, env=env, creationflags=creationflags)
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if hard_stop and stop_evt is not None and stop_evt.is_set():
                try:
                    if os.name == 'nt' and hasattr(signal, 'CTRL_BREAK_EVENT'):
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        proc.terminate()
                except Exception:
                    pass
                # Give it a moment; then kill if still alive
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                break
            time.sleep(0.2)
    finally:
        rc = proc.poll()

    if rc:
        raise subprocess.CalledProcessError(rc, cmd)
    metrics = compute_metrics_from_csv(out_csv)
    return out_csv, metrics


def main(parquet_folder: str, main_script_rel: str | None = None,
         out_results: str = 'DOE_Strategy_1/DOE_results.csv', limit: int | None = 30,
         jobs: int = 2,
         max_specs: int | None = None,
         families: list[str] | None = None,
         ema: list[str] | None = None,
         step: int = 10,
         stop_file: str | None = None):
    # Resolve paths robustly regardless of current working directory
    script_dir = os.path.dirname(__file__)
    pkg_root = os.path.dirname(script_dir)  # .../DOE_Strategy_1
    repo_root = os.path.dirname(pkg_root)

    # Determine main script path
    if main_script_rel is None:
        main_script_path = os.path.join(pkg_root, 'Backtest_HBullD_parquet_with_brokerage_opt_slope.py')
    elif os.path.isabs(main_script_rel):
        main_script_path = main_script_rel
    else:
        # Try relative to package root first, then repo root
        cand1 = os.path.join(pkg_root, main_script_rel)
        cand2 = os.path.join(repo_root, main_script_rel)
        main_script_path = cand1 if os.path.exists(cand1) else cand2

    parquet_folder = os.path.abspath(parquet_folder)
    out_results = os.path.abspath(out_results)
    out_dir = os.path.join(os.path.dirname(out_results), 'DOE_runs')
    # Developer-friendly defaults via env flag (only if user didn't override)
    dev_flag = os.environ.get('DOE_DEV', '').lower() in ('1', 'true', 'yes', 'y')
    if dev_flag:
        # Apply conservative defaults only if not explicitly set
        limit = limit if limit is not None and limit != 30 else 2
        families = families if families else ['hb_any']
        ema = ema if ema else ['v1']
        step = step if step != 10 else 20

    # Hard stop is always enabled by design

    # Config priority: if config.doe exists (default DOE_Strategy_1/config.doe), use it
    default_cfg = os.path.join(pkg_root, 'config.doe')
    cfg_path = os.environ.get('DOE_CONFIG') or (default_cfg if os.path.exists(default_cfg) else None)
    if cfg_path and os.path.exists(cfg_path):
        print(f"Using DOE config: {cfg_path}")
        specs = build_specs_from_config(cfg_path)
    else:
        specs = build_specs(
            limit=limit,
            max_specs=max_specs,
            families=families,
            ema_variants=ema,
            step=step,
        )

    # Print DOE parameter-space stats
    def _stats(spec_list: list[dict]) -> str:
        if not spec_list:
            return "No specs"
        def tup(x):
            return tuple(x) if isinstance(x, (list, tuple)) else x
        ema_set = sorted({s.get('ema_variant') for s in spec_list})
        fam_set = sorted({s.get('div_family') for s in spec_list})
        ll_set = sorted({tup(s.get('rsi_lower_low_range')) for s in spec_list})
        hl_set = sorted({tup(s.get('rsi_higher_low_range')) for s in spec_list})
        gap_set = sorted({tup(s.get('date_gap_range')) for s in spec_list})
        slope_set = sorted({tup(s.get('slope_range')) for s in spec_list})
        lines = [
            f"EMA variants: {len(ema_set)} -> {ema_set}",
            f"Families: {len(fam_set)} -> {fam_set}",
            f"RSI_LL pairs: {len(ll_set)}",
            f"RSI_HL pairs: {len(hl_set)}",
            f"DateGap pairs: {len(gap_set)}",
            f"Slope pairs: {len(slope_set)}",
            f"Total specs: {len(spec_list)}",
        ]
        return " | ".join(lines)

    print("DOE parameter space:", _stats(specs))

    # Determine jobs from env if provided; otherwise use all CPU cores
    try:
        env_jobs = int(os.environ.get('DOE_JOBS')) if os.environ.get('DOE_JOBS') else None
    except Exception:
        env_jobs = None
    if isinstance(env_jobs, int) and env_jobs > 0:
        jobs = env_jobs
    elif jobs == 2:  # default from CLI; take all cores by default
        try:
            cpu = os.cpu_count() or 1
            jobs = max(1, cpu)
        except Exception:
            jobs = max(jobs, 1)

    total = len(specs)
    print(f"DOE: running {total} spec(s) with {jobs} parallel job(s)")

    # To avoid inner oversubscription, set default inner procs if not provided
    if not os.environ.get('DOE_LOAD_PROCS'):
        try:
            cpu = os.cpu_count() or 1
            inner = max(1, cpu // max(jobs, 1))
        except Exception:
            inner = 1
        os.environ['DOE_LOAD_PROCS'] = str(inner)
        print(f"Inner parquet-load procs per backtest: {inner} (DOE_LOAD_PROCS)")

    # Stop controls: Ctrl+C and stop-file support
    stop_evt = threading.Event()

    def _handle_signal(signum, frame):
        print(f"Received signal {signum}; stopping after current tasks...")
        stop_evt.set()

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass

    # Resolve stop-file path (env takes precedence)
    stop_file = os.environ.get('DOE_STOP_FILE') or stop_file
    if not stop_file:
        stop_file = os.path.join(os.path.dirname(out_results), 'STOP')

    rows = []
    submitted = 0
    in_flight = {}
    start_ts = time.time()
    completed = 0
    last_eta_print = 0.0

    def should_stop():
        return stop_evt.is_set() or (stop_file and os.path.exists(stop_file))

    # Submit-and-drain pattern so we can honor stop requests
    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as exe:
        try:
            # Initial fill up to concurrency
            while submitted < total and len(in_flight) < max(jobs, 1):
                if should_stop():
                    break
                spec = specs[submitted]
                idx = submitted + 1
                fut = exe.submit(run_backtest, main_script_path, parquet_folder, spec, out_dir, stop_evt, True)
                in_flight[fut] = (idx, spec)
                submitted += 1
                print(f"[queued {idx}/{total}] {spec['ema_variant']} | {spec['div_family']} | RSI_LL {tuple(spec['rsi_lower_low_range'])} | RSI_HL {tuple(spec['rsi_higher_low_range'])}")

            # Process completions and keep submitting until stop or all specs queued
            while in_flight:
                # Use short timeout to periodically check stop conditions
                done_any = False
                try:
                    for fut in as_completed(list(in_flight.keys()), timeout=0.5):
                        done_any = True
                        i, spec = in_flight.pop(fut)
                        try:
                            csv_path, m = fut.result()
                            row = {
                                'i': i,
                                'ema_variant': spec['ema_variant'],
                                'div_family': spec['div_family'],
                                'rsi_ll_range': tuple(spec['rsi_lower_low_range']),
                                'rsi_hl_range': tuple(spec['rsi_higher_low_range']),
                                'date_gap_range': tuple(spec['date_gap_range']),
                                'slope_range': tuple(spec['slope_range']),
                                **m,
                                'csv_path': csv_path,
                            }
                            rows.append(row)
                            completed += 1
                            elapsed = time.time() - start_ts
                            avg = (elapsed / completed) if completed else 0.0
                            remaining = max(total - completed, 0)
                            eta = remaining * avg
                            print(f"[done {i}/{total}] trades={m.get('trades')} ret={m.get('total_return_pct'):.2f}% mdd={m.get('max_drawdown_pct'):.2f}% | progress {completed}/{total} elapsed {elapsed:0.1f}s eta {eta:0.1f}s")
                        except Exception as e:
                            rows.append({
                                'i': i,
                                'ema_variant': spec['ema_variant'],
                                'div_family': spec['div_family'],
                                'error': f'run failed: {e}',
                            })
                            completed += 1
                            elapsed = time.time() - start_ts
                            remaining = max(total - completed, 0)
                            eta = (elapsed / completed) * remaining if completed else 0.0
                            print(f"[fail {i}/{total}] {e} | progress {completed}/{total} elapsed {elapsed:0.1f}s eta {eta:0.1f}s")

                        # After each completion, try to submit a new one if allowed
                        while submitted < total and len(in_flight) < max(jobs, 1) and not should_stop():
                            spec2 = specs[submitted]
                            idx2 = submitted + 1
                            fut2 = exe.submit(run_backtest, main_script_path, parquet_folder, spec2, out_dir, stop_evt, True)
                            in_flight[fut2] = (idx2, spec2)
                            submitted += 1
                            print(f"[queued {idx2}/{total}] {spec2['ema_variant']} | {spec2['div_family']} | RSI_LL {tuple(spec2['rsi_lower_low_range'])} | RSI_HL {tuple(spec2['rsi_higher_low_range'])}")
                except TimeoutError:
                    # Periodic heartbeat with ETA while waiting
                    now = time.time()
                    if now - last_eta_print >= 5.0 and completed:
                        elapsed = now - start_ts
                        remaining = max(total - completed, 0)
                        avg = elapsed / completed
                        eta = remaining * avg
                        print(f"[progress] {completed}/{total} done, {len(in_flight)} running, elapsed {elapsed:0.1f}s, eta {eta:0.1f}s")
                        last_eta_print = now

                if should_stop():
                    print("Stop requested; waiting for running tasks to finish...")
                    break

        except KeyboardInterrupt:
            print("KeyboardInterrupt: stopping after current tasks...")
            stop_evt.set()
        finally:
            # Prevent new tasks; optionally cancel queued futures
            # Signal stop to running tasks so they can hard-stop their children
            stop_evt.set()
            try:
                exe.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass

    # Write partial or full results
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_results), exist_ok=True)
    df.to_csv(out_results, index=False)
    return out_results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True, help='Folder containing parquet files')
    p.add_argument('--limit', type=int, default=30, help='Limit number of DOE specs to run')
    p.add_argument('--main', default=None, help='Path to main backtest script (defaults to DOE_Strategy_1/Backtest_HBullD_parquet_with_brokerage_opt_slope.py)')
    p.add_argument('--jobs', type=int, default=2, help='Parallel jobs (subprocess concurrency)')
    p.add_argument('--out', default='DOE_Strategy_1/DOE_results.csv', help='Output CSV for DOE summary')
    p.add_argument('--max-specs', type=int, default=None, help='Global cap across all generated specs (overrides per-combo totals)')
    p.add_argument('--families', default=None, help='Comma-separated divergence families to include (e.g. hb_any,cb_gen)')
    p.add_argument('--ema', default=None, help='Comma-separated EMA variants to include (e.g. v1,v2)')
    p.add_argument('--step', type=int, default=10, help='Grid step for (x,y) RSI/gap/slope ranges (default 10)')
    p.add_argument('--stop-file', default=None, help='Path to a sentinel file; if it exists, runner stops scheduling new specs')
    # Hard stop always enabled; no CLI flag needed
    p.add_argument('--config', default=None, help='Path to config.doe (TOML/JSON). If present, takes priority over CLI flags')
    args = p.parse_args()
    families = [s.strip() for s in args.families.split(',')] if args.families else None
    ema = [s.strip() for s in args.ema.split(',')] if args.ema else None
    # Allow explicit config via CLI to override autodiscovery
    if args.config:
        os.environ['DOE_CONFIG'] = args.config
    res = main(
        args.folder,
        main_script_rel=args.main,
        out_results=args.out,
        limit=args.limit,
        jobs=args.jobs,
        max_specs=args.max_specs,
        families=families,
        ema=ema,
        step=args.step,
        stop_file=args.stop_file,
    )
    print(f"DOE summary written to: {res}")

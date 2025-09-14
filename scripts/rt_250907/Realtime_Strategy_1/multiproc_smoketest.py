# multiproc_smoketest.py
from multiprocessing import get_context, cpu_count

def f(n: int) -> int:
    return n*n

if __name__ == "__main__":             # wichtig unter Windows (spawn)
    ctx = get_context("spawn")
    with ctx.Pool(processes=cpu_count()) as pool:
        out = pool.map(f, range(10), chunksize=2)
    print(out)
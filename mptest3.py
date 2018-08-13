import multiprocessing
from multiprocessing.pool import Pool

n_workers = multiprocessing.cpu_count()
print(n_workers)
import multiprocessing as mp
import random
import string

random.seed(123)

def cube(x):
    return x**3

# Define an output queue
output = mp.Queue()

# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
                        string.ascii_lowercase
                        + string.ascii_uppercase
                        + string.digits)
                   for i in range(length))
    output.put(rand_str)

def mainfunc():
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(cube, args=(x,)) for x in range(1, 7)]
    output = [p.get() for p in results]
    print(output)

if __name__ == '__main__':
    mainfunc()
import multiprocessing as mp
from multiprocessing import Process, Lock
from multiprocessing import Pool
import random
import string
from time import sleep
import os

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

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def fname(name):
    info('function fname')
    print('hello', name)
    sleep(10)
    print('{} slept for {} seconds'.format( name,10))

def mainfunc():
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(cube, args=(x,)) for x in range(1, 7)]
    output = [p.get() for p in results]
    print(output)

def f(l, i):
    # l.acquire()
    try:
        print('hello world', i)
    finally:
        # l.release()
        pass

def print_result(result):
    print (result)

def f1(x):
    # sleep(0.5)
    # global glob
    # glob *= x
    # print("in process {} and result is {}".format(x,glob) )
    counter = 0
    for i in range(100000001):
        counter += 1

    print (counter + x)
    return counter

def callf1():
    with Pool(processes=4) as pool:
        # print "[0, 1, 4,..., 81]"
        # print(pool.map(f1, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f1, range(12)):
            print(i)

        # evaluate "f(10)" asynchronously
        # res = pool.apply_async(f1, args=([10]))
        #print(res.get(timeout=1))  # prints "100"

        # make worker sleep for 10 secs
        # res = pool.apply_async(sleep, [10])
        # print(res.get(timeout=1))

def f2():
    info('main line')
    p = Process(target=fname, args=('bob',))
    p.start()
    p2 = Process(target=fname, args=('alice',))
    p2.start()
    # p.join()
    #p2.join()

if __name__ == '__main__':
    # mainfunc()
    #glob = 2
    #callf1()
    f2()

    '''
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()

    '''
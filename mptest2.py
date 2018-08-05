"""
We want to run a function asychronously and run a callback function with multiple parameters when it
returns! In this example, we are pretending we're analyzing
the names and ages of some people. We want to print out:
jack 0
jill 1
james 2
"""

import time
from multiprocessing import Pool
import numpy as np
import random
from operator import itemgetter
import copy

def sum(task, a, b):
    sleepTime = random.randint(1, 8)
    print(task, " requires ", sleepTime, " seconds to finish")
    time.sleep(sleepTime)
    return a+b,task

def printResult(result):
    global mystack
    tmp = mystack
    value,task = result
    mystack += value
    print("task {} before is {} and after is {}".format(task,tmp,mystack))


def async_function(name):
    """
    Function we want to run asynchronously and in parallel,
    usually one with heavy input/output, though using a
    dummy function here.
    """
    time.sleep(3)
    print("the name in function is {} ".format(name))
    return name

def callback_function(name):
    """
    Function we want to run with the result of the async
    function. The async function returns one parameter, but
    this function takes two parameters. We have to figure
    out how to pass the age parameter from the async function
    to this function..
    """
    age = 10
    print ("name is {} and age is {}".format(name,age))


if __name__ == '__main__':
    myPool = Pool(processes=3)
    mystack = 0

    for i in range(1,6):
        print("to call task when mystack is {}".format(mystack))
        if i%3 == 0 :
            myPool.close()
            myPool.join()
            myPool = Pool(processes=3)
        result1 = myPool.apply_async(sum, args=("task{}".format(i), 10*i, mystack,), callback=printResult)


    print("Submitted tasks to pool")

    myPool.close()
    myPool.join()
Author: Nick Tapp-Hughes (nic98@live.unc.edu)
Last Updated: Nov 16, 2021

This folder contains all the python code for the menthol smoking model.
This is tested on my computer which is running Ubuntu linux. 
So this code might not work on Windows but should work on MacOS.

There are two types of python files:
    1. .py files which contain a python program that you can run.

    2. .ipynb files which are interactive python notebooks
        comprising of "cells" of code which can be run
        independently (one at a time). The python runtime state
        is the same over all the cells, so each cell knows about
        the variables, functions, classes, etc. in the other cells.
        Useful for testing out ideas and statistical probing of data.

Here is a description of relevant files:

main.py 
    This is the driver code which starts the simulation. 
    You can specify the number of replications you would like by typing
    'python main.py <num simulations>' in the command line, 
    where <num simulations> should be an integer.
    All this code does is load the path data, beta values, and life tables,
    then it starts the simulation using the simulation.py code

simulation.py
    This contains the simulation.
    There is one class, called Simulation, which has
    many attributes and one method, called simulate().
    The attributes of Simulation describe the relevant parameters of the simulation.
    The simulate() method does roughly the following:
        1. formats the data in a suitable way for fast calculations
            (using the indicators that were used in logistic regression)
        2. loops over the years of the simulation, doing the following for each year:
            a. write data to record-keeping structures for later output
            b. incorporate life tables to randomly determine death
            c. update the smoking status of people via logistic regression equations
                i. obtain logits
                ii. convert logits to probabilities
                iii. sample probabilities to obtain outcome
            d. keep track of who has smoked in their lifetime
               so that they can never become a 'never smoker'
        3. saves the records of the simulation as a file on disk

life_model.ipynb
    This is a python notebook that I was using just to test out some ideas
    and play around with the data. 

inspect_data.ipynb
    I used this python notebook to load some data that was output by
    a single simulation replication and make some plots of that data.

aggregate_data.ipynb
    I used this python notebook to aggregate data over many simulation runs
    and to plot some resuls from many replications

data_stats.ipynb
    I plan to use this notebook to obtain statistics and tables from the
    simulation output.
    
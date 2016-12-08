Need to run 'conda update numba' or equivalent, eg. 'pip install numba --upgrade' in order to run the code, 
as some of the functions used in the jit decorated functions are only available in newer versions of numba 
than the one currently installed by default on the college computers.

Might need to update scipy in a manner similar to the above, I found that on the college computers, 
it might be required to remove scipy beforehand using 'conda remove scipy'
before being able to update it via 'conda install scipy' or 'conda update scipy'
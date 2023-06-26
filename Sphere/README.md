This folder contains python code for LSEF 
on the sphere.

The folder "Sphere" must be your current working directory
(so that imports work correctly).

You can run script "nn_train.py" and then "Main.py" 
to run the simulation once. Or you can run the simulation 
multiple times with the script "mult_times_script.py". 
In this case the script will run the script "nn_train.py";
then it will run "Main.py" multiple times 
and save the results.

You can run the multiple simulation with different values
of the parameters. The variable "params_for_cycles" 
(which is list) contains tuples of length 3:
(parameter name, list of parameter values, 
type of parameter variable). For example: 

[('e_size', [5, 10, 20, 40, 80], 'int'),]

The number of simulations is controlled by the list of 
random seeds "all_seeds.txt" 
(integer numbers with whitespaces between them).
When the current simulation ends, the corresponding seed 
is written to the file "all seeds <parameter_name> 
<parameter_value>.txt". If the system crushes or 
the script is interrupted, then you 
can run the script again; it will start with the current 
seed (the script checks which seeds are finished). 
If you need to run everything from the beginning, 
you can delete the seeds written by the script.

Each simulation creates a subfolder named 
"<date_time> main my_seed_mult <seed_number> 
<parameter_name> <parameter_value>" in the folder 
"images". After all the simulations are over you should
move or copy all these folders to a folder and specify
its name in the variable "source_folder" in the script 
"Results.py". The variable "params_for_cycles" can be 
the same as in the previous script, or it can be its 
sublist. Then run "Results.py" to get the final results.




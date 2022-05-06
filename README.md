# MoE-method
Instructions to run the experiments
This document is prepared to facilitate the reproduction of this study. 
Detailed steps to run the experiments.
Put all the datasets files on .csv format in the folder Datasets. These files with .csv extension should be in folder 'Datasets' in same directory.
Remove the column names (header) from the datasets. 
Dataset should not contain any attribute names. Last column should have class labels. Label should numeric 0/1. 0 shows non-faulty and 1 shows faulty.
All the .py files (Wrapper, Tester, and helper) should be in the same directory. 
Put the datasets names in the dataset_name [] list before running wrapper file. 
Run the Wrapper.py file. It will call tester.py and helper.py implicitly.
After successful execution of wrapper.py file, results will be stored automatically in the results folder with the label of file name_expert_method.

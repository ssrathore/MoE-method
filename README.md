# MoE-method
Instructions to run the experiments
This document is prepared to facilitate the reproduction of this study. 
Detailed steps to run the experiments.
1. Put all the datasets files on .csv format in the folder Datasets. These files with .csv extension should be in folder 'Datasets' in same directory.
2. Remove the column names (header) from the datasets. 
3. Dataset should not contain any attribute names. Last column should have class labels. Label should numeric 0/1. 0 shows non-faulty and 1 shows faulty.
4. All the .py files (Wrapper, Tester, and helper) should be in the same directory. 
5. Put the datasets names in the dataset_name [] list before running wrapper file. 
6. Run the Wrapper.py file. It will call tester.py and helper.py implicitly.
7. After successful execution of wrapper.py file, results will be stored automatically in the results folder with the label of file name_expert_method.

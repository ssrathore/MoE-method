import os

cwd = os.getcwd()

dataset_name = ["ant-1.7","camel-1.2","camel-1.4","camel-1.6","CM1","JDT_Core","PDE_UI",
                "Equinox_Framework","ivy-2.0","jedit-4.3","JM1","KC1","KC2","KC3","Lucene",
                "mylyn","PC1","poi-3.0","prop-1","synapse-1.2","velocity-1.6",
                "xalan-2.7","xerces-1.4","eclipse-2.0","eclipse-2.1","eclipse-3.0",
                "MC1","MC2","MW1","PC2","PC3","PC4","PC5","prop-2","prop-3",
                "prop-4","prop-5","prop-6","xalan-2.4","xalan-2.5","xalan-2.6"]
                

for dtname in dataset_name:
    cmd = 'python '+cwd+ '\DS_desc.py ' + dtname
    print("=================="+cmd)
    os.system(cmd)

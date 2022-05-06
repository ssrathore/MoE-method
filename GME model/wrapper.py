import os
#import tester as test

cwd = os.getcwd()

#These files with .csv extension should be in folder 'Datasets' in same directory.
#Dataset should not contain attribute names.
#Last column should have class labels.
#Label should numeric 0/1
#0 shows non faulty
dataset_name = ["Ant-1.7","Camel-1.2","Camel-1.4","camel-1.6","ivy-2.0","jedit-4.3","poi-3.0","synapse-1.2","velocity-1.6","xalan-2.4","xalan-2.5","xalan-2.6","xalan-2.7",                 "xerces-1.4", "CM1", "eclipse-2.0","eclipse-2.1","eclipse-3.0", "Equinox_Framework","JDT_Core","JM1","KC1","KC2","KC3","Lucene","MC1","MC2","MW1","mylyn","PC1", "PC2","PC3","PC4","PC5", "hbase-0.94.0","hive-0.9.0","jruby-1.1","wicket-1.3.0-beta2", 
                "prop-1","prop-2", "prop-2","prop-3","prop-4","prop-5","prop-6",
                "ivy-2.01", "synapse-1.2","velocity-1.6","xalan-2.4","xalan-2.5","xalan-2.6","xalan-2.7","xerces-1.4"
                ]
print(dataset_name)

#list_DS = [0, 0.2, 0.4, 0.6, 0.8, 1]    #Data Selection threshold
#list_LP = [0, 0.2, 0.4, 0.6, 0.8, 1]    #Label Prediction threshold

list_DS = [0.2,0.4]
list_LP = [0.6,0.8]    #Label Prediction threshold

#DT = Decision Tree
#MLP = Multi-layer Perceeptron
expert_model = ["DT","NB","LR","MLP"]
                    #Base model

#indi = Individual/single model
#bag = Bagging
#ME = Mixture of Experts
agg_type = ["indi", "bag", "ME","boost", "stack"]
                #Aggregation/ensemble type

for d in dataset_name:
    for e in expert_model:
        for a in agg_type:
            for t1 in list_LP:
                for t2 in list_DS:
                    cmd = 'python '+cwd+ '\\tester.py ' + d +' '+ e +' '+ a +' '+ str(t1) +' '+ str(t2)
                    print(cmd)
                    os.system(cmd)
                    

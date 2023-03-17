# Random Forest Library
## Open Source Random Forest Library for Educational Purposes. <br>
This library was coded from scratch for the second assignment of the 'Data Mining' course at Istanbul Technical University. In order to work fast, data flow has been tried to be provided via numpy as much as possible. If you want to have an idea and learn about how the decision tree and random forest algorithms are implemented, it is recommended to review the files. Not recommended for professional use!


## How to use?
```python
from random_forest import RandomForest 
rf = RandomForest(forest_size=1)
rf.bootstrapping(XX) #XX = dataset with X's and y's. XX must be pandas.DataFrame
rf.fitting()
y_pred = rf.majority_voting(X) # X = dataset. X must be pandas.DataFrame
```

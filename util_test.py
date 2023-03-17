#Author: @mserdarnazli(turkchen)


import unittest
import util
import decision_tree as Dc
import csv
import ast
import time
import numpy as np

class TestUtil(unittest.TestCase):
    def test_entropy(self):
        pass
        self.assertEqual(util.entropy(np.array([0, 0, 0, 1, 1, 1])), 1)

    def test_information_gain(self):
        pass
        l1 = np.array([0, 0, 0, 1, 1, 1])
        l2_1 = np.array([0,0])
        l2_2 = np.array([0,1,1,1])
        l2 = [l2_1,l2_2]
        self.assertAlmostEqual(util.information_gain(l1,l2),0.4691)

    def test_DecisionTree(self):
        X = [[12,3,4,5],
             [20,2,2,4],
             [5,3,1,6],
             [16,5,6,4],
             [1,2,10,20],
             [12,4,45,6],
             [-10,22,3,10]]
        y = [0,1,1,0,1,2,5]
        start_time = time.time()
        X,y,_ = self.read_dataset()
        print(f"Reading took: ----{time.time()-start_time} secs-------")
        test_X = X[:100000]
        test_y = y[:100000]

        X = X[:10000]
        y = y[:10000]
        dtree = Dc.DecisionTree()

        start_time = time.time()
        dtree.train(X,y)
        print(f"Train took: ----{time.time() - start_time} secs-------")

        start_time = time.time()
        predicted = dtree.classify(test_X)
        print(f"prediction took: ----{time.time() - start_time} secs-------")

        #print("predicted: ", predicted[:10], "\n", "real: ", y[:10])

        start_time = time.time()
        conf_matrix = util.confusion_matrix_(predicted, test_y)
        acc,recall,prec,f1 = util.eval_metrics(conf_matrix)
        print(f"Eval results: \nAcc:{acc}\nRecall:{recall}\nprec:{prec}\nf1:{f1}")
        print(f"Evalution results took: ----{time.time() - start_time} secs-------")

    def read_dataset(self):
        X = list()
        y = list()
        XX = list()  # Contains data features and data labels

        # Loading data set
        print("...Reading Airline Passenger Satisfaction Dataset...")
        with open("airline_passenger_satisfaction.csv") as f:
            next(f, None)
            for line in csv.reader(f, delimiter=","):
                xline = []
                for i in range(len(line)):
                    xline.append(ast.literal_eval(line[i]))

                X.append(xline[:-1])
                y.append(xline[-1])
                XX.append(xline[:])
        return X, y, XX


if __name__ == '__main__':
    unittest.main()

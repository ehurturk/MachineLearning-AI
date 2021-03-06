import os
import json

class Logger:
    def __init__(self):
        self.__dir = os.path.dirname(__file__)
        self.__filename = os.path.join(self.__dir, "log.json")
        self.data = []
        self.message = [] # change later



    def log(self, total_iter, fcost, frsq):
        self.message.append({"Total Iterations": total_iter, "Final Cost": fcost, "Final R Squared:": frsq})
        self.data.append(self.message)

        with open(self.__filename, "w") as out:
            json.dump(self.data, out)

    def msg(self, params, iter_n, cost, rsq):
        with open(self.__filename, "r") as jsondata:
            try:
                self.data = json.load(jsondata)
            except json.JSONDecodeError:
                self.data = []

        self.message.append({
                "Iteration": iter_n,
                "Cost": cost,
                "R Squared": rsq,
                "Parameters": list(params),

        })







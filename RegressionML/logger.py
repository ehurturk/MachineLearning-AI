import os
import json
import datetime

class Logger:
    def __init__(self):
        self.__dir = os.path.dirname(__file__)
        self.__filename = os.path.join(self.__dir, "log.json")
        self.logm = {}
        self.date = datetime.datetime.now().strftime("%m/%d/%Y")
        self.logm[self.date] = []

    def log(self, total_iter):
        self.logm[self.date].append({"Total Iterations": total_iter})
        with open(self.__filename, "a") as out:
            json.dump(self.logm, out)

    def msg(self, params, iter_n, cost):
        self.logm[self.date].append({
                "Iteration": iter_n,
                "Cost": cost,
                "Parameters": list(params),
        })





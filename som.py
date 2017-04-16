#!/usr/bin/env python
from mvpa2.suite import SimpleSOMMapper
import numpy as np
import random
import json
import matplotlib.pyplot as plt


def print_countries(countries):
    for el in countries:
        print("Country: ", el)
        for good in countries[el]:
            print(good)
        print("")


def parse_data(data):
    countries = dict()
    for el in data["data"]:
        if el["country"] in countries:
            prev = list()
            lst = countries[el["country"]]
            prev.extend(lst)
            prev.append([el["item"], el["Value"]])
            countries.update({el["country"]: prev})
        else:
            countries.update({el["country"]: [[el["item"], el["Value"]]]})
    return countries


def normalize(countries):
    names = list()
    res = list()
    for el in countries:
        s = 0
        lst = list()
        for good in countries[el]:
            if good[1] is None:
                s += 0
            else:
                s += good[1]
        for good in countries[el]:
            if good[1] is None:
                good[1] = 0
            else:
                if s != 0.0:
                    good[1] = good[1] / s
            lst.append(good[1])
        res.append(lst)
        names.append(el)
    return (res, names)


def countries_in_clusters(x, y, mapped, names):
    match = [[list()] * x] * y
    i = 0
    for el in mapped:
        if len(match[el[0]][el[1]]) == 0:
            match[el[0]][el[1]] = list([names[i]])
        else:
            match[el[0]][el[1]].append(names[i])
        i += 1
    return match


def print_clusters(x, y, match):
    for yy in range(0, y):
        for xx in range(0, x):
            print(yy, xx)
            print(match[yy][xx])


def visualize(x, y, mapped):
    matrix = np.zeros((y, x))
    for dot in mapped:
        y = dot[0]
        x = dot[1]
        matrix[y][x] += 1
    plt.matshow(matrix, cmap=plt.cm.YlGn)
    plt.colorbar()
    plt.show()


data_file = open("FAOTSJUL2016_2.json", "r")
data = json.load(data_file)
countries = dict()

countries = parse_data(data)
print_countries(countries)
res = normalize(countries)
data_sheet = res[0]
names = res[1]
tr_data_sheet = list()

for i in range(0, 50):
    tr_data_sheet.append(random.choice(data_sheet))

y = 7
x = 7
som = SimpleSOMMapper((y, x), 400, learning_rate=0.05)
som.train(np.array(tr_data_sheet))

mapped = som(data_sheet)
match = countries_in_clusters(x, y, mapped, names)
print_clusters(x, y, match)
visualize(x, y, mapped)

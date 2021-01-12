from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups


# Question 1
twenty_train = fetch_20newsgroups(subset='train')

doc_number = list(map(lambda category: len(fetch_20newsgroups(
    subset='train', categories=[category]).data), twenty_train.target_names))

plt.figure(figsize=(10, 10))
plt.bar(["Cat" + str(i+1) for i in range(len(doc_number))], doc_number)
plt.savefig("q1.png")

# Question 2

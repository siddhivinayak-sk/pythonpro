from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

"""
Seaborn usages
Seaborn is a library that uses Matplotlib underneath to plot graphs.
"""


def main():
    normal_distribution()
    binomial_distribution()
    normal_vs_binomial_distribution()
    poisson_distribution()
    poisson_vs_binomial_distribution()
    uniform_distribution()
    logistic_distribution()
    normal_vs_logistic_distribution()
    multinomial_distribution()
    exponential_distribution()
    chi_square_distribution()
    rayleigh_distribution()
    rayleigh_vs_chi_square_distribution()
    pareto_distribution()
    zipf_distribution()
    pass


def zipf_distribution():
    """
    Zipf Distribution is used to sample data based on zipf's law.
    Zipf's Law: In a collection the nth common term is 1/n times of the most common term.
    E.g. 5th common word in english has 1/5th the frequency of the most used word.
    It has two parameters:
    a - distribution parameter.
    size - The shape of the returned array.
    """
    data = random.zipf(a=2, size=(2, 3))
    seaborn_plot(data)
    pass


def pareto_distribution():
    """
    A distribution following Pareto's law i.e. 80-20 distribution (20% factors cause 80% outcome).
    It has two parameters:
    a - shape parameter.
    size - The shape of the returned array.
    """
    data = random.pareto(a=2, size=(2, 3))
    seaborn_plot(data)
    pass


def rayleigh_vs_chi_square_distribution():
    """
    Relation: At unit stddev and 2 degrees of freedom rayleigh and chi square represent the same distributions.
    """
    sns.distplot(random.rayleigh(scale=1, size=1000), hist=False, label='rayleigh')
    sns.distplot(random.chisquare(df=2, size=1000), hist=False, label='chi square')
    plt.show()
    pass


def rayleigh_distribution():
    """
    Rayleigh Distribution is used in signal processing.
    It has two parameters:
    scale - (standard deviation) decides how flat the distribution will be default 1.0.
    size - The shape of the returned array.
    """
    data = random.rayleigh(scale=2, size=(2, 3))
    seaborn_plot(data)
    pass


def chi_square_distribution():
    """
    Chi Square Distribution is used as a basis to verify the hypothesis.
    It has two parameters:
    df - (degree of freedom).
    size - The shape of the returned array.
    """
    data = random.chisquare(df=2, size=(2, 3))
    seaborn_plot(data)
    pass


def exponential_distribution():
    """
    Exponential Distribution is used for describing time till next event e.g. failure/success etc.
    Relation: Poisson distribution deals with number of occurences of an event in a time period whereas exponential distribution deals with the time between these events.
    It has two parameters:
    scale - inverse of rate ( see lam in poisson distribution ) defaults to 1.0.
    size - The shape of the returned array.
    """
    data = random.exponential(scale=2, size=(2, 3))
    seaborn_plot(data)
    pass


def multinomial_distribution():
    """
    Multinomial Distribution is a generalization of binomial distribution.
    It describes outcomes of multi-nomial scenarios unlike binomial where scenarios must be only one of two.
    e.g. Blood type of a population, dice roll outcome.
    It has three parameters:
    n - number of possible outcomes (e.g. 6 for dice roll).
    pvals - list of probabilties of outcomes (e.g. [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] for dice roll).
    size - The shape of the returned array.
    """
    data = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
    seaborn_plot(data)
    pass


def normal_vs_logistic_distribution():
    """
    Both distributions are near identical, but logistic distribution has more area under the tails, meaning it represents more possibility of occurrence of an event further away from mean.
    For higher value of scale (standard deviation) the normal and logistic distributions are near identical apart from the peak.
    """
    sns.distplot(random.normal(loc=0, scale=1, size=1000), hist=False, label='normal')
    sns.distplot(random.logistic(loc=0, scale=1, size=1000), hist=False, label='logistic')
    plt.show()
    pass


def logistic_distribution():
    """
    Logistic Distribution is used to describe growth.
    eg. The bacteria grows rapidly in the beginning but then declines.
    It is used in Neural networks.
    loc - mean, where the peak is. Default 0.
    scale - standard deviation, the flatness of distribution. Default 1.
    size - The shape of the returned array.
    """
    data = random.logistic(loc=1, scale=2, size=1000)
    seaborn_plot(data)
    pass


def uniform_distribution():
    """
    Uniform Distribution is a distribution where a random variable is equally likely to take any value.
    It is also known as Rectangular Distribution.
    """
    data = random.uniform(low=0, high=1, size=1000)
    seaborn_plot(data)
    pass


def poisson_vs_binomial_distribution():
    """
    Difference between Poisson and Binomial Distribution:
    Poisson Distribution is used when the number of occurences is theoretically infinite.
    Binomial Distribution is used when number of occurences is exactly finite.
    eg. A dice is rolled 6 times. What is probability of getting 1 for 1 time? This is Binomial.
    eg. A dice is rolled infinite times. What is probability of getting 1 for 1 time? This is Poisson.
    With the increase in number of trials, the binomial distribution tends towards the poisson distribution.
    """
    sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
    sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')
    plt.show()
    pass


def poisson_distribution():
    """
    Poisson Distribution is a Discrete Distribution.
    It estimates how many times an event can happen in a specified time.
    eg. If someone eats twice a day what is probability he will eat thrice?
    lam - rate or known number of occurences e.g. 2 for above problem.
    size - The shape of the returned array.
    """
    data = random.poisson(lam=2, size=10)
    seaborn_plot(data)
    pass


def normal_vs_binomial_distribution():
    """
    Difference between Normal and Binomial Distribution:
    Normal Distribution is continous whereas Binomial is discrete.
    Normal Distribution is symmetrical whereas Binomial is not.
    For Normal Distribution, mean=median=mode but for Binomial, mean=median=mode=(n*p)
    With higher n, Binomial Distribution approches Normal Distribution.
    """
    sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
    sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')
    plt.show()
    pass


def binomial_distribution():
    """
    Binomial Distribution is a Discrete Distribution.
    It describes the outcome of binary scenarios, eg. toss of a coin, it will either be head or tails.
    n - number of trials.
    p - probability of occurence of each trial (e.g. for toss of a coin 0.5 each).
    size - The shape of the returned array.
    """
    data = random.binomial(n=10, p=0.5, size=1000)
    seaborn_plot(data)
    pass


def normal_distribution():
    """
    Also called Gaussian Distribution. Its curve is called Bell Curve.
    It fits the probability distribution of many events, eg. IQ Scores, Heartbeat etc.
    It has three parameters:
    loc - (Mean) where the peak of the bell exists.
    scale - (Standard Deviation) how flat the graph distribution should be.
    size - The shape of the returned array.
    """
    data = random.normal(loc=1, scale=2, size=(2, 3))
    seaborn_plot(data)
    pass


def seaborn_plot(data):
    sns.distplot(data, hist=True)  # Plot histogram
    plt.show()
    pass


main()

import csv


if __name__ == '__main__':
    with open('winequality-red.csv', 'r') as f:
        wines = list(csv.reader(f, delimiter=';'))

    qualities = [float(item[-1]) for item in wines[1:]]
    print sum(qualities) / len(qualities)

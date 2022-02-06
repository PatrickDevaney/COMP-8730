# https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/
# Ahmed Fawzy Gad
# AI/ML engineer and a talented technical writer who authors 4 scientific books and more than 80 articles and tutorials. https://www.linkedin.com/in/ahmedfgad

import numpy
def levenshtein(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))
    for t1 in range(len(token1) + 1): distances[t1][0] = t1
    for t2 in range(len(token2) + 1): distances[0][t2] = t2
    a = 0; b = 0; c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
              a = distances[t1][t2 - 1]
              b = distances[t1 - 1][t2]
              c = distances[t1 - 1][t2 - 1]

              if (a <= b and a <= c): distances[t1][t2] = a + 1
              elif (b <= a and b <= c): distances[t1][t2] = b + 1
              else: 
                if (token1[t1 - 1] == token2[t2 - 1]): distances[t1][t2] = c
                else: distances[t1][t2] = c + 2

    #print_matrix(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]

def print_matrix(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(str(int(distances[t1][t2])).zfill(2), end=" ")
        print()

#levenshtein("application", "aplikatiion")

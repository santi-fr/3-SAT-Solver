
"""
import regex, sys
import numpy as np

from sympy import *

def parseSAT(expression):
     Parses a given SAT expression in CNF to its variables.
    ###Args:
        #- expression (string): The expression as a string to be parsed.
    ###Returns:
        #A list of lists of the variables in the expression. 
    
    expression = expression.replace('(', '').replace(')', '')
    newExpression = regex.split(' *A *', expression)
    newExpression = [regex.split(' *V *', x) for x in newExpression]
    return newExpression



def simplifyVariable(variable):
    if variable[0] == '¬':
        return "(1 - " + variable[1:] + ")"
    return variable

def transform(list):
    i = 1
    expression = "-("
    for x, y, z in list:
        x, y, z = simplifyVariable(x), simplifyVariable(y), simplifyVariable(z)
        expression += "(1 + w" + str(i) + ")*(" + x + " + " + y + " + " + z + ") - " + x + "*" + y + " - " + x + "*" + z + " - " + y + "*" + z + " - 2*w"+str(i)
        if len(list) != i:
            expression += " + "
        i += 1
    expression += ")"
    return expression

def SATToQUBO():
    expression = "(x1 V x2 V x3) A (¬x1 V x2 V x3) A (x1 V ¬x2 V x3) A (¬x1 V x2 V ¬x3)"
    expression = transform(parseSAT(expression))
    #expression = "-" + expression
    x1, x2, x3, w1, w2, w3, w4 = symbols('x1 x2 x3 w1 w2 w3 w4')

    # diccionario de sustitución
    substitutions = {'x1': x1, 'x2': x2, 'x3': x3, 'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4}

    # reemplazar las variables en la expresión original
    for var, sym in substitutions.items():
        expression = expression.replace(var, str(sym))# diccionario de sustitución
    substitutions = {'x1': x1, 'x2': x2, 'x3': x3, 'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4}

    # reemplazar las variables en la expresión original
    for var, sym in substitutions.items():
        expression = expression.replace(var, str(sym))
    sympy_expression = sympify(expression)
    simplified_expression = simplify(sympy_expression)
    print(simplified_expression)

def QUBOtoMatrix():

    # Set N to be the maximum value of the QUBO variables
    N = 3

    # Create an NxN matrix with all elements initialized to 0
    matrix = np.zeros((N, N))

    # Set the corresponding matrix element to the coefficient of each term
    matrix[0, 1] = -1  # x1x2
    matrix[1, 2] = 2   # 2x2x3
    matrix[0, 2] = -3  # -3x1x3

    # Since the matrix is symmetric, set the element at [i, j] equal to the element at [j, i]
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())

    print(matrix)
"""
import dwavebinarycsp
import numpy as np
from dwave.system import LeapHybridSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

def sat_to_qubo(clauses, num_variables):
    qubo = {}
    # initialize QUBO with all possible keys
    for i in range(1, num_variables + 1):
        for j in range(i, num_variables + 1):
            qubo[(i, j)] = 0
            qubo[(j, i)] = 0

    # add clauses
    for clause in clauses:
        q1, q2, q3 = clause

        # add q1
        if q1 < 0:
            qubo[(abs(q1), abs(q1))] -= 1
        else:
            qubo[(abs(q1), abs(q1))] += 1

        # add q2
        if q2 < 0:
            qubo[(abs(q2), abs(q2))] -= 1
        else:
            qubo[(abs(q2), abs(q2))] += 1

        # add q3
        if q3 < 0:
            qubo[(abs(q3), abs(q3))] -= 1
        else:
            qubo[(abs(q3), abs(q3))] += 1

        # add interactions
        qubo[(abs(q1), abs(q2))] += 1
        qubo[(abs(q1), abs(q3))] += 1
        qubo[(abs(q2), abs(q3))] += 1

    return qubo


def interpret_sample(sample_set):
    variables = list(sample_set.variables)
    samples = sample_set.record
    sample_list = []
    for sample in samples:
        sample_dict = dict(zip(variables, sample))
        sample_list.append([(-1) ** sample_dict[var] for var in variables])
    return sample_list






from dwave.system import LeapHybridSampler
if __name__ == '__main__':
    # define a 3-SAT problem instance
    num_variables = 3
    clauses = [[1, -2, 3], [2, 3, -1]]

    # convert 3-SAT to QUBO
    Q = sat_to_qubo(clauses, num_variables)

    # solve QUBO using D-Wave quantum computer
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q, num_reads=10)

    # print the optimal solution
    samples = response.record.sample
    energies = response.record.energy
    for idx in range(len(samples)):
        solution = samples[idx]
        energy = energies[idx]
        variable_values = interpret_sample(solution)
        print(f"Solution {idx+1}: {variable_values}")
        print(f"Energy: {energy}\n")






import dwavebinarycsp
import numpy as np
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


def interpret_sample(sample):
    return [(-1) ** sample[var] for var in range(len(sample))]


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

        # extract variable values from solution
        var_vals = {}
        for var in range(1, num_variables+1):
            if var in solution:
                var_vals[var] = solution[var]
            else:
                var_vals[var] = -1 * solution[-var]

        # print the values of the variables that minimize the energy
        print("Variable values:")
        for var, val in var_vals.items():
            print(f"x{var} = {val}")



    












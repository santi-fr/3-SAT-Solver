import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

def generate_3_SAT_problem(num_vars, num_clauses):
    # Generate a random 3-SAT problem
    clauses = [[random.randint(1, num_vars) * random.choice([-1, 1]) for _ in range(3)] for _ in range(num_clauses)]
    return clauses

def run_performance_tests(start_dim, end_dim, start_ratio_var, end_ratio_var, num_tests=10):
    # Run performance tests
    performance_data = np.zeros(((end_dim - start_dim) // 10 + 1, int((end_ratio_var - start_ratio_var) / 0.5) + 1))

    # Iterate over the number of clauses and ratio of variables
    for dim_index, dim in enumerate(range(start_dim, end_dim + 1, 10)):
        for ratio_var_index, ratio_var in enumerate(np.arange(start_ratio_var, end_ratio_var + 0.5, 0.5)):
            times = []
            # Run the test num_tests times
            for _ in range(num_tests):
                num_vars = int(dim * ratio_var)
                num_clauses = dim
                clauses = generate_3_SAT_problem(num_vars, num_clauses)

                start_time = time.time()
                bqm, variables = build_bqm(clauses)
                sampler = EmbeddingComposite(DWaveSampler())
                sampleset = sampler.sample(bqm, num_reads=50)
                end_time = time.time()

                execution_time = end_time - start_time
                times.append(execution_time)

            avg_time = np.mean(times)
            performance_data[dim_index, ratio_var_index] = avg_time
            print(f"Number of Clauses: {dim}, Ratio of Variables: {ratio_var}, Avg. Time: {avg_time}")

    return performance_data


def plot_performance(performance_data, start_dim, end_dim, start_ratio_var, end_ratio_var, filename):
    # Plot the performance data
    fig, ax = plt.subplots()

    ratio_vars = np.arange(start_ratio_var, end_ratio_var + 0.5, 0.5)
    num_clauses_list = list(range(start_dim, end_dim + 1, 10))

    for dim, dim_performance in zip(num_clauses_list, performance_data):
        ax.plot(ratio_vars, dim_performance, label=f"Num. Clauses {dim}")

    ax.set_xlabel("Ratio of Variables")
    ax.set_ylabel("Average Execution Time (s)")
    ax.set_title("3-SAT Solver Performance")
    ax.legend()

    plt.savefig(filename)
    plt.show()


# Function to build the binary quadratic model (BQM) for the given 3-SAT problem
# Function to build the binary quadratic model (BQM) for the given 3-SAT problem
def build_bqm(clauses):
    
    # Function to convert a clause to penalty terms
    def clause_to_penalty_terms(variables, clause):
        penalty_terms = []
        for var in clause:
            if var > 0:
                penalty_terms.append(variables[var])
            else:
                penalty_terms.append(1 - variables[-var])

        return penalty_terms

    # Create a new BQM with binary variables
    bqm = BinaryQuadraticModel('BINARY')

    # Calculate the highest variable number (max_var) in the problem
    max_var = max(abs(lit) for clause in clauses for lit in clause)

    # Add variables to the BQM, creating a dictionary mapping variable numbers to BQM variables
    variables = {i: bqm.add_variable(i) for i in range(1, max_var + 1)}

    # Add terms to the BQM for each clause
    for clause in clauses:
        # Convert the clause to penalty terms
        penalty_terms = clause_to_penalty_terms(variables, clause)
        
        # Add interactions between the penalty terms in the BQM if they are distinct
        if penalty_terms[0] != penalty_terms[1]:
            bqm.add_interaction(penalty_terms[0], penalty_terms[1], 1)
        if penalty_terms[0] != penalty_terms[2]:
            bqm.add_interaction(penalty_terms[0], penalty_terms[2], 1)
        if penalty_terms[1] != penalty_terms[2]:
            bqm.add_interaction(penalty_terms[1], penalty_terms[2], 1)
        
        # Increment the BQM's offset by 1
        bqm.offset += 1

    return bqm, variables


# Function to check if a given solution is valid for the given clauses
def is_solution_valid(solution, clauses):
    for clause in clauses:
        if not any([solution[abs(literal)] == (literal > 0) for literal in clause]):
            return False
    return True

# Function to test the 3-SAT solver with different test cases 
def test_3sat_solver():
    test_cases = [
        {
            "clauses": [[1, 2, 3], [1, -2, -3], [-1, 2, -3]],
            "description": "Test case 1"
        },
        {
            "clauses": [[1, 2, 3], [1, -2, -3], [-1, 2, -3], [-1, -2, 3]],
            "description": "Test case 2"
        },
        {
            "clauses": [[1, 2, -3], [-1, -2, 3], [-1, 2, 3], [1, -2, 3]],
            "description": "Test case 3"
        },
    ]

    for test_case in test_cases:
        print(f"Running {test_case['description']}")
        bqm, _ = build_bqm(test_case['clauses'])
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample(bqm, num_reads=100)  # Increased num_reads to 100
        sample = sampleset.lowest().first.sample  # Get the lowest energy sample
        solution = {abs(key): value for key, value in sample.items() if key > 0}

        if is_solution_valid(solution, test_case['clauses']):
            print("Solution is valid.")
        else:
            print("Solution is invalid.")
        print()


if __name__ == '__main__':
    # Set up argument parser and add new arguments for performance mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run test cases')
    parser.add_argument("--performance", action="store_true", help="Measure performance for a range of dimensions and ratios")
    parser.add_argument("--dims", nargs=2, type=int, metavar=("DIM_START", "DIM_END"), help="Range of dimensions to test")
    parser.add_argument("--ratio", nargs=2, type=int, metavar=("RATIO_START", "RATIO_END"), help="Range of ratios to test")
    args = parser.parse_args()

    if args.test:
        # If --test flag is present, run the test_3sat_solver function
        test_3sat_solver()
    elif args.performance:
        # If --performance flag is present, run the performance mode
        start_dim, end_dim = args.dims
        start_ratio_var, end_ratio_var = args.ratio
        performance_data = run_performance_tests(start_dim, end_dim, start_ratio_var, end_ratio_var)
        plot_performance(performance_data, start_dim, end_dim, start_ratio_var, end_ratio_var, "3sat_performance.png")
    else:
        # Set up scenario
        clauses = [[1, 2, 3], [1, -2, -3], [-1, 2, -3]]

        # Build BQM
        bqm, variables = build_bqm(clauses)

        # Run on DWave sampler
        print("\nRunning D-Wave sampler...")
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample(bqm, num_reads=1000)

        # Find the lowest energy solution
        lowest_energy_solution = sampleset.lowest().first.sample

        # Print the results
        print("\nSolution:")
        for var in variables:
            print(f"{var}: {lowest_energy_solution[int(var)]}")


import argparse
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

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
        
        # Add interactions between the penalty terms in the BQM
        bqm.add_interaction(penalty_terms[0], penalty_terms[1], 1)
        bqm.add_interaction(penalty_terms[0], penalty_terms[2], 1)
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
        sampleset = sampler.sample(bqm, num_reads=1000)  # Increased num_reads to 1000
        sample = sampleset.lowest().first.sample  # Get the lowest energy sample
        solution = {abs(key): value for key, value in sample.items() if key > 0}

        if is_solution_valid(solution, test_case['clauses']):
            print("Solution is valid.")
        else:
            print("Solution is invalid.")
        print()


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run test cases')
    args = parser.parse_args()

    if args.test:
        # If --test flag is present, run the test_3sat_solver function
        test_3sat_solver()
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


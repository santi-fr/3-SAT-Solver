# 3-SAT Solver using D-Wave Quantum Annealer

This program solves 3-SAT problems using the D-Wave quantum annealer. It takes a set of clauses and returns a solution that satisfies the problem if one exists.

## Prerequisites

- Python 3.7 or higher
- D-Wave Ocean SDK
- Access to a D-Wave quantum annealer

## Installation

1. Install the D-Wave Ocean SDK by following the instructions in the [official documentation](https://docs.ocean.dwavesys.com/en/latest/overview/install.html).
2. Set up your D-Wave API token by following the instructions in the [official documentation](https://docs.ocean.dwavesys.com/en/latest/overview/dwave_cloud.html).

## Usage

There are two modes for the program:

1. Test mode: In this mode, the program runs a set of predefined test cases.
2. Regular mode: In this mode, the program solves a predefined 3-SAT problem.

### Test Mode

To run the program in test mode, execute the following command:

  ```python SAT-Solver.py --test```
  
This will run a set of predefined test cases to validate the 3-SAT solver.

### Regular Mode

To run the program in regular mode, execute the following command:

 ```python SAT-Solver.py```


This will solve a predefined 3-SAT problem and output the solution.

## Modifying the 3-SAT Problem

To solve a different 3-SAT problem, modify the `clauses` variable in the `else` block within the `__main__` section of the script. The `clauses` variable should be a list of lists, where each inner list represents a clause with three literals.

For example:

  ```clauses = [[1, 2, 3], [1, -2, -3], [-1, 2, -3]]```

### License

This project is licensed under the MIT License. See the LICENSE file for details.

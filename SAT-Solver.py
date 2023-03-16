import regex, sys
from sympy import *

def parseSAT(expression):
    """ Parses a given SAT expression in CNF to its variables.
    ###Args:
        - expression (string): The expression as a string to be parsed.
    ###Returns:
        A list of lists of the variables in the expression. 
    """
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

SATToQUBO()
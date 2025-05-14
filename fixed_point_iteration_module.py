import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    """
    Fixed Point Iteration Method for solving x = g(x)
    
    Parameters:
    -----------
    g : function
        The function in the form x = g(x)
    x0 : float
        Initial guess
    tol : float, optional
        Tolerance for convergence (default is 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default is 100)
    
    Returns:
    --------
    x_values : list
        List of all approximations
    iterations : int
        Number of iterations performed
    converged : bool
        Boolean indicating if method converged within tolerance
    """
    x_values = [x0]
    
    # Iteration loop
    for i in range(max_iter):
        # Calculate next approximation
        x_new = g(x_values[-1])
        x_values.append(x_new)
        
        # Check for convergence
        if abs(x_new - x_values[-2]) < tol:
            return x_values, i+1, True
    
    # If max iterations reached without convergence
    return x_values, max_iter, False

def create_function_from_expression(expr_str):
    """
    Create a numerical function from a string expression
    
    Parameters:
    -----------
    expr_str : str
        String representation of a mathematical expression
    
    Returns:
    --------
    function
        A function that evaluates the expression numerically
    """
    # Setup transformation for parsing
    transformations = standard_transformations + (implicit_multiplication_application,)
    
    # Create symbolic expression
    x = sp.Symbol('x')
    expr = parse_expr(expr_str, transformations=transformations)
    
    # Create numerical function
    return sp.lambdify(x, expr, modules="numpy")

def plot_convergence(x_values, actual_root=None):
    """
    Plot the convergence of the fixed point iteration method
    
    Parameters:
    -----------
    x_values : list
        List of approximations
    actual_root : float, optional
        The actual root (if known)
    """
    iterations = range(len(x_values))
    plt.figure(figsize=(10, 6))
    
    # Plot approximations
    plt.plot(iterations, x_values, 'bo-', label='Approximations')
    
    # Plot actual root if known
    if actual_root is not None:
        plt.axhline(y=actual_root, color='r', linestyle='--', label=f'Actual Root: {actual_root}')
    
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Approximation')
    plt.title('Convergence of Fixed Point Iteration Method')
    plt.legend()
    
    # Plot error if actual root is known
    if actual_root is not None:
        errors = [abs(x - actual_root) for x in x_values]
        plt.figure(figsize=(10, 6))
        plt.semilogy(iterations, errors, 'ro-', label='Error')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Error (log scale)')
        plt.title('Error Convergence - Fixed Point Iteration')
        plt.legend()
    
    plt.show()

def analyze_convergence(g, x_star, delta=0.01):
    """
    Analyze the convergence of the fixed-point iteration by examining |g'(x*)|
    
    Parameters:
    -----------
    g : function
        The function in the form x = g(x)
    x_star : float
        The fixed point (solution)
    delta : float
        Small value for numerical differentiation
    
    Returns:
    --------
    derivative : float
        Approximate value of g'(x*)
    convergence_status : str
        Description of convergence behavior
    """
    # Approximate the derivative using central difference
    derivative = (g(x_star + delta) - g(x_star - delta)) / (2 * delta)
    abs_derivative = abs(derivative)
    
    if abs_derivative < 1:
        return derivative, "Convergent (|g'(x*)| < 1)"
    elif abs_derivative > 1:
        return derivative, "Divergent (|g'(x*)| > 1)"
    else:
        return derivative, "Indeterminate (|g'(x*)| = 1)"

def parse_and_rearrange_equation(equation_str):
    """
    Parse an equation and rearrange it to the form x = g(x)
    
    Parameters:
    -----------
    equation_str : str
        String representation of an equation (f(x) = 0 or y = f(x))
    
    Returns:
    --------
    rhs_expr : str
        Right-hand side expression of the rearranged equation x = g(x)
    f_expr : str
        Original function f(x) where f(x) = 0 is the equation to solve
    """
    try:
        # Ensure equation has the format "y = ..." or "... = 0"
        if "=" not in equation_str:
            equation_str = equation_str + " = 0"
        
        # Split the equation into left and right parts
        parts = equation_str.split("=", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        
        # Convert to the form f(x) = 0
        if rhs == "0":
            f_expr = lhs
        else:
            f_expr = f"({lhs}) - ({rhs})"
        
        # Create symbolic expression for analysis
        transformations = standard_transformations + (implicit_multiplication_application,)
        x = sp.Symbol('x')
        f_sym = parse_expr(f_expr, transformations=transformations)
        
        # Try to solve for x algebraically
        try:
            # For equations with specific patterns like our example cos(x) - 3x + 1 = 0
            # we can identify the coefficient of x
            x_coef = sp.diff(f_sym, x).subs([(sp.sin(x), 0), (sp.cos(x), 0)])
            
            # If x has a coefficient, we can rearrange to isolate x
            if x_coef != 0:
                # Extract terms without x
                other_terms = f_sym - x_coef * x
                # Rearrange to x = g(x)
                g_expr = -other_terms / x_coef
                rhs_expr = str(g_expr)
                return rhs_expr, f_expr
            
        except Exception:
            pass
        
        # If automated algebraic rearrangement fails, use a default rearrangement
        # For example, if we have f(x) = 0, we'll try x = x - f(x)
        rhs_expr = f"x - ({f_expr})"
        
        return rhs_expr, f_expr
    
    except Exception as e:
        print(f"Error parsing equation: {e}")
        return None, None

def run_fixed_point_iteration(equation=None, x0=0.5, tol=1e-6, max_iter=100):
    """
    Run fixed point iteration based on equation and parameters
    
    Parameters:
    -----------
    equation : str, optional
        Equation to solve
    x0 : float, optional
        Initial guess
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
        
    Returns:
    --------
    result_dict : dict
        Dictionary containing results and analysis
    """
    if equation is None:
        equation = "cos(x) - x = 0"
    
    # Parse and rearrange the equation
    rhs_expr, f_expr = parse_and_rearrange_equation(equation)
    if rhs_expr is None:
        return {"error": "Failed to parse the equation. Please try again with proper syntax."}
    
    # Create the iteration function g(x)
    try:
        g = create_function_from_expression(rhs_expr)
        f = create_function_from_expression(f_expr)
    except Exception as e:
        return {"error": f"Error creating function: {e}"}
    
    # Run the fixed point iteration
    x_values, iterations, converged = fixed_point_iteration(g, x0, tol, max_iter)
    
    # Create result dictionary
    result = {
        "original_equation": f"{f_expr} = 0",
        "fixed_point_form": f"x = {rhs_expr}",
        "initial_guess": x0,
        "final_approximation": x_values[-1],
        "function_value": f(x_values[-1]),
        "iterations": iterations,
        "converged": converged,
        "x_values": x_values,
        "iteration_errors": [abs(x_values[i] - x_values[i-1]) for i in range(1, len(x_values))],
        "function_values": [f(x) for x in x_values]
    }
    
    # Analyze convergence if method converged
    if converged:
        derivative, convergence_status = analyze_convergence(g, x_values[-1])
        result["derivative_abs"] = abs(derivative)
        result["convergence_status"] = convergence_status
    
    return result

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    """
    Fixed Point Iteration Method for solving x = g(x)
    
    Parameters:
    -----------
    g : function
        The function in the form x = g(x)
    x0 : float
        Initial guess
    tol : float, optional
        Tolerance for convergence (default is 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default is 100)
    
    Returns:
    --------
    x_values : list
        List of all approximations
    iterations : int
        Number of iterations performed
    converged : bool
        Boolean indicating if method converged within tolerance
    """
    x_values = [x0]
    
    # Iteration loop
    for i in range(max_iter):
        # Calculate next approximation
        x_new = g(x_values[-1])
        x_values.append(x_new)
        
        # Check for convergence
        if abs(x_new - x_values[-2]) < tol:
            return x_values, i+1, True
    
    # If max iterations reached without convergence
    return x_values, max_iter, False

def create_function_from_expression(expr_str):
    """
    Create a numerical function from a string expression
    
    Parameters:
    -----------
    expr_str : str
        String representation of a mathematical expression
    
    Returns:
    --------
    function
        A function that evaluates the expression numerically
    """
    # Setup transformation for parsing
    transformations = standard_transformations + (implicit_multiplication_application,)
    
    # Create symbolic expression
    x = sp.Symbol('x')
    expr = parse_expr(expr_str, transformations=transformations)
    
    # Create numerical function
    return sp.lambdify(x, expr, modules="numpy")

def plot_convergence(x_values, actual_root=None):
    """
    Plot the convergence of the fixed point iteration method
    
    Parameters:
    -----------
    x_values : list
        List of approximations
    actual_root : float, optional
        The actual root (if known)
    """
    iterations = range(len(x_values))
    plt.figure(figsize=(10, 6))
    
    # Plot approximations
    plt.plot(iterations, x_values, 'bo-', label='Approximations')
    
    # Plot actual root if known
    if actual_root is not None:
        plt.axhline(y=actual_root, color='r', linestyle='--', label=f'Actual Root: {actual_root}')
    
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Approximation')
    plt.title('Convergence of Fixed Point Iteration Method')
    plt.legend()
    
    # Plot error if actual root is known
    if actual_root is not None:
        errors = [abs(x - actual_root) for x in x_values]
        plt.figure(figsize=(10, 6))
        plt.semilogy(iterations, errors, 'ro-', label='Error')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Error (log scale)')
        plt.title('Error Convergence - Fixed Point Iteration')
        plt.legend()
    
    plt.show()

def analyze_convergence(g, x_star, delta=0.01):
    """
    Analyze the convergence of the fixed-point iteration by examining |g'(x*)|
    
    Parameters:
    -----------
    g : function
        The function in the form x = g(x)
    x_star : float
        The fixed point (solution)
    delta : float
        Small value for numerical differentiation
    
    Returns:
    --------
    derivative : float
        Approximate value of g'(x*)
    convergence_status : str
        Description of convergence behavior
    """
    # Approximate the derivative using central difference
    derivative = (g(x_star + delta) - g(x_star - delta)) / (2 * delta)
    abs_derivative = abs(derivative)
    
    if abs_derivative < 1:
        return derivative, "Convergent (|g'(x*)| < 1)"
    elif abs_derivative > 1:
        return derivative, "Divergent (|g'(x*)| > 1)"
    else:
        return derivative, "Indeterminate (|g'(x*)| = 1)"

def parse_and_rearrange_equation(equation_str):
    """
    Parse an equation and rearrange it to the form x = g(x)
    
    Parameters:
    -----------
    equation_str : str
        String representation of an equation (f(x) = 0 or y = f(x))
    
    Returns:
    --------
    rhs_expr : str
        Right-hand side expression of the rearranged equation x = g(x)
    f_expr : str
        Original function f(x) where f(x) = 0 is the equation to solve
    """
    try:
        # Ensure equation has the format "y = ..." or "... = 0"
        if "=" not in equation_str:
            equation_str = equation_str + " = 0"
        
        # Handle "y = f(x)" case by replacing it with "f(x) = 0"
        if equation_str.startswith("y ="):
            equation_str = equation_str[4:] + " = 0"
        
        # Split the equation into left and right parts
        parts = equation_str.split("=", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        
        # Convert to the form f(x) = 0
        if rhs == "0":
            f_expr = lhs
        else:
            f_expr = f"({lhs}) - ({rhs})"
        
        # Create symbolic expression for analysis
        transformations = standard_transformations + (implicit_multiplication_application,)
        x = sp.Symbol('x')
        f_sym = parse_expr(f_expr, transformations=transformations)
        
        # Try to solve for x algebraically
        try:
            # For equations with specific patterns like our example cos(x) - 3x + 1 = 0
            # we can identify the coefficient of x
            x_coef = sp.diff(f_sym, x).subs([(sp.sin(x), 0), (sp.cos(x), 0)])
            
            # If x has a coefficient, we can rearrange to isolate x
            if x_coef != 0:
                # Extract terms without x
                other_terms = f_sym - x_coef * x
                # Rearrange to x = g(x)
                g_expr = -other_terms / x_coef
                rhs_expr = str(g_expr)
                return rhs_expr, f_expr
            
        except Exception:
            pass
        
        # If automated algebraic rearrangement fails, use a default rearrangement
        # For example, if we have f(x) = 0, we'll try x = x - f(x)
        rhs_expr = f"x - ({f_expr})"
        
        return rhs_expr, f_expr
    
    except Exception as e:
        print(f"Error parsing equation: {e}")
        return None, None

def run_fixed_point_iteration(equation=None, x0=0.5, tol=1e-6, max_iter=100):
    """
    Run fixed point iteration based on equation and parameters
    
    Parameters:
    -----------
    equation : str, optional
        Equation to solve
    x0 : float, optional
        Initial guess
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
        
    Returns:
    --------
    result_dict : dict
        Dictionary containing results and analysis
    """
    if equation is None:
        equation = "cos(x) - x = 0"
    
    # Parse and rearrange the equation
    rhs_expr, f_expr = parse_and_rearrange_equation(equation)
    if rhs_expr is None:
        return {"error": "Failed to parse the equation. Please try again with proper syntax."}
    
    # Create the iteration function g(x)
    try:
        g = create_function_from_expression(rhs_expr)
        f = create_function_from_expression(f_expr)
    except Exception as e:
        return {"error": f"Error creating function: {e}"}
    
    # Run the fixed point iteration
    x_values, iterations, converged = fixed_point_iteration(g, x0, tol, max_iter)
    
    # Create result dictionary
    result = {
        "original_equation": f"{f_expr} = 0",
        "fixed_point_form": f"x = {rhs_expr}",
        "initial_guess": x0,
        "final_approximation": x_values[-1],
        "function_value": f(x_values[-1]),
        "iterations": iterations,
        "converged": converged,
        "x_values": x_values,
        "iteration_errors": [abs(x_values[i] - x_values[i-1]) for i in range(1, len(x_values))],
        "function_values": [f(x) for x in x_values]
    }
    
    # Analyze convergence if method converged
    if converged:
        derivative, convergence_status = analyze_convergence(g, x_values[-1])
        result["derivative_abs"] = abs(derivative)
        result["convergence_status"] = convergence_status
    
    return result


# import numpy as np
# import matplotlib.pyplot as plt
# import sympy as sp
# from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
#     """
#     Fixed Point Iteration Method for solving x = g(x)
    
#     Parameters:
#     -----------
#     g : function
#         The function in the form x = g(x)
#     x0 : float
#         Initial guess
#     tol : float, optional
#         Tolerance for convergence (default is 1e-6)
#     max_iter : int, optional
#         Maximum number of iterations (default is 100)
    
#     Returns:
#     --------
#     x_values : list
#         List of all approximations
#     iterations : int
#         Number of iterations performed
#     converged : bool
#         Boolean indicating if method converged within tolerance
#     """
#     x_values = [x0]
    
#     # Iteration loop
#     for i in range(max_iter):
#         # Calculate next approximation
#         x_new = g(x_values[-1])
#         x_values.append(x_new)
        
#         # Check for convergence
#         if abs(x_new - x_values[-2]) < tol:
#             return x_values, i+1, True
    
#     # If max iterations reached without convergence
#     return x_values, max_iter, False

# def create_function_from_expression(expr_str):
#     """
#     Create a numerical function from a string expression
    
#     Parameters:
#     -----------
#     expr_str : str
#         String representation of a mathematical expression
    
#     Returns:
#     --------
#     function
#         A function that evaluates the expression numerically
#     """
#     # Setup transformation for parsing
#     transformations = standard_transformations + (implicit_multiplication_application,)
    
#     # Create symbolic expression
#     x = sp.Symbol('x')
#     expr = parse_expr(expr_str, transformations=transformations)
    
#     # Create numerical function
#     return sp.lambdify(x, expr, modules="numpy")

# def plot_convergence(x_values, actual_root=None):
#     """
#     Plot the convergence of the fixed point iteration method
    
#     Parameters:
#     -----------
#     x_values : list
#         List of approximations
#     actual_root : float, optional
#         The actual root (if known)
#     """
#     iterations = range(len(x_values))
#     plt.figure(figsize=(10, 6))
    
#     # Plot approximations
#     plt.plot(iterations, x_values, 'bo-', label='Approximations')
    
#     # Plot actual root if known
#     if actual_root is not None:
#         plt.axhline(y=actual_root, color='r', linestyle='--', label=f'Actual Root: {actual_root}')
    
#     plt.grid(True)
#     plt.xlabel('Iteration')
#     plt.ylabel('Approximation')
#     plt.title('Convergence of Fixed Point Iteration Method')
#     plt.legend()
    
#     # Plot error if actual root is known
#     if actual_root is not None:
#         errors = [abs(x - actual_root) for x in x_values]
#         plt.figure(figsize=(10, 6))
#         plt.semilogy(iterations, errors, 'ro-', label='Error')
#         plt.grid(True)
#         plt.xlabel('Iteration')
#         plt.ylabel('Error (log scale)')
#         plt.title('Error Convergence - Fixed Point Iteration')
#         plt.legend()
    
#     plt.show()

# def analyze_convergence(g, x_star, delta=0.01):
#     """
#     Analyze the convergence of the fixed-point iteration by examining |g'(x*)|
    
#     Parameters:
#     -----------
#     g : function
#         The function in the form x = g(x)
#     x_star : float
#         The fixed point (solution)
#     delta : float
#         Small value for numerical differentiation
    
#     Returns:
#     --------
#     derivative : float
#         Approximate value of g'(x*)
#     convergence_status : str
#         Description of convergence behavior
#     """
#     # Approximate the derivative using central difference
#     derivative = (g(x_star + delta) - g(x_star - delta)) / (2 * delta)
#     abs_derivative = abs(derivative)
    
#     if abs_derivative < 1:
#         return derivative, "Convergent (|g'(x*)| < 1)"
#     elif abs_derivative > 1:
#         return derivative, "Divergent (|g'(x*)| > 1)"
#     else:
#         return derivative, "Indeterminate (|g'(x*)| = 1)"

# def parse_and_rearrange_equation(equation_str):
#     """
#     Parse an equation and rearrange it to the form x = g(x)
    
#     Parameters:
#     -----------
#     equation_str : str
#         String representation of an equation (f(x) = 0 or y = f(x))
    
#     Returns:
#     --------
#     rhs_expr : str
#         Right-hand side expression of the rearranged equation x = g(x)
#     f_expr : str
#         Original function f(x) where f(x) = 0 is the equation to solve
#     """
#     try:
#         # Ensure equation has the format "y = ..." or "... = 0"
#         if "=" not in equation_str:
#             equation_str = equation_str + " = 0"
        
#         # Handle "y = f(x)" case by converting it to "f(x) = 0"
#         if equation_str.startswith("y ="):
#             equation_str = equation_str[4:] + " = 0"
        
#         # Split the equation into left and right parts
#         parts = equation_str.split("=", 1)
#         lhs = parts[0].strip()
#         rhs = parts[1].strip()
        
#         # Convert to the form f(x) = 0
#         if rhs == "0":
#             f_expr = lhs
#         else:
#             f_expr = f"({lhs}) - ({rhs})"
        
#         # Create symbolic expression for analysis
#         transformations = standard_transformations + (implicit_multiplication_application,)
#         x = sp.Symbol('x')
#         f_sym = parse_expr(f_expr, transformations=transformations)
        
#         # For equations of the form a*x + b*cos(x) + c = 0,
#         # we'll extract the coefficient of x to create the fixed point form
        
#         # Try to extract coefficient of x
#         try:
#             # Differentiate to get coefficient of x
#             diff_result = sp.diff(f_sym, x)
            
#             # Try to extract linear term coefficient
#             x_coef = None
#             for term in diff_result.as_ordered_terms():
#                 if term.has(x) and term.is_polynomial(x) and sp.degree(term, x) == 0:
#                     x_coef = term
#                     break
                
#             # If we couldn't find it that way, try direct substitution approach
#             if x_coef is None:
#                 # Get coefficient by removing trigonometric functions
#                 x_coef = diff_result.subs([(sp.sin(x), 0), (sp.cos(x), 1)])
                
#                 # If this still contains x, try another approach
#                 if x_coef.has(x):
#                     test_point = 0  # Choose a point where cos(x)=1, sin(x)=0
#                     # Evaluate the derivative at x=0 and subtract any contribution from trig functions
#                     deriv_at_zero = diff_result.subs(x, test_point)
#                     trig_contrib = diff_result.subs([(x, test_point), (sp.cos(x), 0), (sp.sin(x), 0)])
#                     x_coef = deriv_at_zero - trig_contrib
            
#             # Check if we have a valid coefficient
#             if x_coef is not None and x_coef != 0:
#                 # Extract all other terms (the ones that don't involve linear x)
#                 terms = f_sym - x_coef * x
                
#                 # Rearrange to x = g(x) form
#                 g_expr = sp.simplify(-terms / x_coef)
                
#                 return str(g_expr), f_expr
#         except Exception as e:
#             print(f"Error extracting coefficient: {e}")
        
#         # If the above approach fails, use a more general method
#         # For equations like cos(x) - 3*x + 1 = 0, solve for x explicitly
#         try:
#             # Attempt to solve for x directly
#             if "cos(x)" in f_expr and "x" in f_expr:
#                 # For forms like cos(x) - a*x + b = 0, use x = (cos(x) + b)/a
#                 # First, collect all coefficients of x
#                 coef_x = sp.diff(f_sym, x).subs([(sp.sin(x), 0), (sp.cos(x), 0)])
                
#                 if coef_x != 0:
#                     # Extract terms without x
#                     others = f_sym - coef_x * x
#                     g_expr = sp.simplify(-others / coef_x)
#                     return str(g_expr), f_expr
#         except Exception as e:
#             print(f"Error in direct solving approach: {e}")
        
#         # Fallback method: standard iteration scheme x = x - f(x)
#         # Note: This may not converge for all equations
#         rhs_expr = f"x - ({f_expr})"
        
#         return rhs_expr, f_expr
    
#     except Exception as e:
#         print(f"Error parsing equation: {e}")
#         return None, None

# def run_fixed_point_iteration(equation=None, x0=0.5, tol=1e-6, max_iter=100):
#     """
#     Run fixed point iteration based on equation and parameters
    
#     Parameters:
#     -----------
#     equation : str, optional
#         Equation to solve
#     x0 : float, optional
#         Initial guess
#     tol : float, optional
#         Tolerance for convergence
#     max_iter : int, optional
#         Maximum number of iterations
        
#     Returns:
#     --------
#     result_dict : dict
#         Dictionary containing results and analysis
#     """
#     if equation is None:
#         equation = "cos(x) - x = 0"
    
#     # Parse and rearrange the equation
#     rhs_expr, f_expr = parse_and_rearrange_equation(equation)
#     if rhs_expr is None:
#         return {"error": "Failed to parse the equation. Please try again with proper syntax."}
    
#     # Create the iteration function g(x)
#     try:
#         g = create_function_from_expression(rhs_expr)
#         f = create_function_from_expression(f_expr)
#     except Exception as e:
#         return {"error": f"Error creating function: {e}"}
    
#     # Run the fixed point iteration
#     x_values, iterations, converged = fixed_point_iteration(g, x0, tol, max_iter)
    
#     # Create result dictionary
#     result = {
#         "original_equation": f"{f_expr} = 0",
#         "fixed_point_form": f"x = {rhs_expr}",
#         "initial_guess": x0,
#         "final_approximation": x_values[-1],
#         "function_value": f(x_values[-1]),
#         "iterations": iterations,
#         "converged": converged,
#         "x_values": x_values,
#         "iteration_errors": [abs(x_values[i] - x_values[i-1]) for i in range(1, len(x_values))],
#         "function_values": [f(x) for x in x_values]
#     }
    
#     # Analyze convergence if method converged
#     if converged:
#         derivative, convergence_status = analyze_convergence(g, x_values[-1])
#         result["derivative_abs"] = abs(derivative)
#         result["convergence_status"] = convergence_status
    
#     return result
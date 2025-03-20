from simplex import SimplexMethod
from scipy.optimize import linprog


# Data for task 1
A = [
  [1, 1, 0, -1],
  [1, -1, 0, 3],
  [2, 1, 1, -1]
]
b = [3, -1, 5]
c = [-1, -1, -1, -5]


# Data for task 2
"""A = [
    [1, 8, 7, -15],
    [1, -5, -6, 11]
]
b = [17, -9]
c = [-4,- 3, -5, 20]"""

simplex = SimplexMethod(A, b, c, _print=True, old_get_pilot=True)
optimal_value, coordinates = simplex.solve()
print()
print("Task 1 (my code):")
print("The optimal solution:", coordinates)
print("The optimal value of the function:", optimal_value)


res1 = linprog(c, A_eq=A, b_eq=b, method='highs')
print("\nResults scipy.optimize.linprog:")
print("The optimal solution:", res1.x)
print("The optimal value of the function:", res1.fun)
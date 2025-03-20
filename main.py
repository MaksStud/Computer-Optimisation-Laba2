from simplex import SimplexMethod
from scipy.optimize import linprog


# Дані для задачі 1
A = [
  [1, 1, 0, -1],
  [1, -1, 0, 3],
  [2, 1, 1, -1]
]
b = [3, -1, 5]
c = [-1, -1, -1, -5]


# Дані для задачі 2
"""A = [
    [1, 8, 7, -15],
    [1, -5, -6, 11]
]
b = [17, -9]
c = [-4,- 3, -5, 20]"""

simplex = SimplexMethod(A, b, c)
optimal_value, coordinates = simplex.solve()
print()
print("Задача 1 (мої код):")
print("Оптимальне рішення:", coordinates)
print("Оптимальне значення функції:", optimal_value)


res1 = linprog(c, A_eq=A, b_eq=b, method='highs')
print("\nРезультати scipy.optimize.linprog:")
print("Оптимальне рішення:", res1.x)
print("Оптимальне значення функції:", res1.fun)
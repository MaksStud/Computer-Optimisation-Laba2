import numpy as np
from tabulate import tabulate


class SimplexMethod:
    def __init__(self, A, b, c, _print=False, old_get_pilot=True):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.num_constraints, self.num_variables = self.A.shape
        self.tableau = self._initialize_tableau()
        self.basis = list(range(self.num_variables, self.num_variables + self.num_constraints))
        self._print = _print
        self.old_get_pilot = old_get_pilot

    def _initialize_tableau(self):
        tableau = np.zeros((self.num_constraints + 1, self.num_variables + self.num_constraints + 1))
        tableau[:-1, :-1] = np.hstack((self.A, np.eye(self.num_constraints)))
        tableau[:-1, -1] = self.b
        tableau[-1, :-1] = np.hstack((self.c, np.zeros(self.num_constraints)))
        return tableau

    def _get_pivot_column(self):
        return np.argmin(self.tableau[-1, :-1])

    def _get_pivot_row_old(self, pivot_col):
        ratios = self.tableau[:-1, -1] / self.tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        return np.argmin(ratios)

    def _get_pivot_row_new(self, pivot_col):
        column = self.tableau[:-1, pivot_col]
        positive_mask = column > 0
        if not np.any(positive_mask):
            raise ValueError("Розв’язок необмежений")
        ratios = np.full_like(column, np.inf, dtype=float)
        ratios[positive_mask] = self.tableau[:-1, -1][positive_mask] / column[positive_mask]
        return np.argmin(ratios)

    def _get_pivot_row(self, pivot_col):
        return self._get_pivot_row_old(pivot_col) if self.old_get_pilot else self._get_pivot_row_new(pivot_col)

    def _pivot(self, pivot_row, pivot_col):
        self._print_pretamble(pivot_col, pivot_row)
        self.tableau[pivot_row, :] /= self.tableau[pivot_row, pivot_col]
        for i in range(len(self.tableau)):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]
        self.basis[pivot_row] = pivot_col
        self._print_after_tamble()

    def solve(self):
        while np.any(self.tableau[-1, :-1] < 0):
            pivot_col = self._get_pivot_column()
            pivot_row = self._get_pivot_row(pivot_col)
            if np.all(self.tableau[:-1, pivot_col] <= 0):
                raise ValueError("Розв’язок необмежений")
            self._pivot(pivot_row, pivot_col)

        optimal_value = -self.tableau[-1, -1]
        solution = self._get_solution()

        alternative_solutions = []
        for j in range(self.num_variables):
            if j not in self.basis and np.isclose(self.tableau[-1, j], 0):
                try:
                    alternative_solution = self._get_alternative_solution(j)
                    alternative_solutions.append(alternative_solution)
                except ValueError:
                    continue

        return optimal_value, solution, alternative_solutions

    def _get_solution(self):
        solution = np.zeros(self.num_variables)
        for i, var_index in enumerate(self.basis):
            if var_index < self.num_variables:
                solution[var_index] = self.tableau[i, -1]
        return solution

    def _print_pretamble(self, pivot_col, pivot_row):
        if self._print:
            print(f"\nОбираємо ведучий стовпець: {pivot_col}, ведучий рядок: {pivot_row}")
            print("Поточна симплекс-таблиця перед обчисленням:")
            self._print_tableau()

    def _print_after_tamble(self):
        if self._print:
            print("Оновлена симплекс-таблиця:")
            self._print_tableau()

    def _print_tableau(self):
        headers = [f"x{i}" for i in range(self.num_variables)] + \
                  [f"s{i}" for i in range(self.num_constraints)] + ["b"]

        row_labels = [f"B{self.basis[i]}" for i in range(self.num_constraints)] + ["Z"]
        table = np.round(self.tableau, 4).tolist()

        print(tabulate(table, headers=headers, showindex=row_labels, tablefmt="fancy_grid"))

    def _get_alternative_solution(self, pivot_col):
        pivot_row = self._get_pivot_row(pivot_col)
        if np.all(self.tableau[:-1, pivot_col] <= 0):
            raise ValueError("Не можна знайти альтернативний розв’язок")

        tableau_copy = self.tableau.copy()
        basis_copy = self.basis.copy()

        self._pivot(pivot_row, pivot_col)
        alternative_solution = self._get_solution()

        self.tableau = tableau_copy
        self.basis = basis_copy

        return alternative_solution

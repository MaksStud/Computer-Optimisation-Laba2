import numpy as np


class SimplexMethod:
    def __init__(self, A, b, c, _print=False):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.num_constraints, self.num_variables = self.A.shape
        self.tableau = self._initialize_tableau()
        self.basis = list(range(self.num_variables, self.num_variables + self.num_constraints))
        self._print = _print

    def _initialize_tableau(self):
        tableau = np.zeros((self.num_constraints + 1, self.num_variables + self.num_constraints + 1))
        tableau[:-1, :-1] = np.hstack((self.A, np.eye(self.num_constraints)))
        tableau[:-1, -1] = self.b
        tableau[-1, :-1] = np.hstack((self.c, np.zeros(self.num_constraints)))
        return tableau

    def _get_pivot_column(self):
        return np.argmin(self.tableau[-1, :-1])

    def _get_pivot_row(self, pivot_col):
        ratios = self.tableau[:-1, -1] / self.tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf  # Уникнення від'ємних або нульових значень
        return np.argmin(ratios)

    #def _get_pivot_row(self, pivot_col):
    #    column = self.tableau[:-1, pivot_col]
    #    positive_mask = column > 0  # Враховуємо лише додатні значення
    #    if not np.any(positive_mask):  # Якщо немає позитивних значень, рішення може бути необмеженим
    #        raise ValueError("Розв’язок необмежений")
    #    ratios = np.full_like(column, np.inf, dtype=float)  # Заповнюємо inf, щоб уникнути помилок
    #    ratios[positive_mask] = self.tableau[:-1, -1][positive_mask] / column[positive_mask]
    #    return np.argmin(ratios)

    def _pivot(self, pivot_row, pivot_col):
        self.__print_pretamble(pivot_col, pivot_row)
        self.tableau[pivot_row, :] /= self.tableau[pivot_row, pivot_col]
        for i in range(len(self.tableau)):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]
        self.basis[pivot_row] = pivot_col
        self.__print_after_tamble()

    def solve(self):
        while np.any(self.tableau[-1, :-1] < 0):
            pivot_col = self._get_pivot_column()
            pivot_row = self._get_pivot_row(pivot_col)
            if np.all(self.tableau[:-1, pivot_col] <= 0):
                raise ValueError("Розв’язок необмежений")
            self._pivot(pivot_row, pivot_col)
        return -self.tableau[-1, -1], self._get_solution()

    def _get_solution(self):
        solution = np.zeros(self.num_variables)
        for i, var_index in enumerate(self.basis):
            if var_index < self.num_variables:
                solution[var_index] = self.tableau[i, -1]
        return solution

    def __print_pretamble(self, pivot_col, pivot_row):
        if self._print:
            print(f"\nОбираємо ведучий стовпець: {pivot_col}, ведучий рядок: {pivot_row}")
            print("Поточна симплекс-таблиця перед обчисленням:")
            print(self.tableau)
            print()

    def __print_after_tamble(self):
        if self._print:
            print("Оновлена симплекс-таблиця:")
            print(self.tableau)

import numpy as np
from tabulate import tabulate


class SimplexMethod:
    """Implementation of the simplex method."""
    def __init__(self, A, b, c, _print=False, old_get_pilot=True):
        """
        Initializes the SimplexMethod instance with given parameters.

        :param A: Coefficient matrix of constraints.
        :param b: Right-hand side values of constraints.
        :param c: Coefficients of the objective function.
        :param _print: Flag to enable verbose output. Defaults to False.
        :param old_get_pilot: Flag to use old pivot row selection method. Defaults to True.
        """
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.num_constraints, self.num_variables = self.A.shape
        self.tableau = self._initialize_tableau()
        self.basis = list(range(self.num_variables, self.num_variables + self.num_constraints))
        self._print = _print
        self.old_get_pilot = old_get_pilot

    def _initialize_tableau(self):
        """
        Constructs the initial simplex tableau.

        Combines A matrix, slack variables, objective function, and constraints into a single matrix.

        :return: Initialized tableau as a numpy array.
        """
        tableau = np.zeros((self.num_constraints + 1, self.num_variables + self.num_constraints + 1))
        tableau[:-1, :-1] = np.hstack((self.A, np.eye(self.num_constraints)))
        tableau[:-1, -1] = self.b
        tableau[-1, :-1] = np.hstack((self.c, np.zeros(self.num_constraints)))
        return tableau

    def _get_pivot_column(self):
        """
        Identifies the entering variable column using the most negative coefficient in the objective row.

        :return: Index of the pivot column.
        """
        return np.argmin(self.tableau[-1, :-1])

    def _get_pivot_row_old(self, pivot_col):
        """
        Selects leaving variable row using minimum ratio test (old method allowing non-positive denominators).

        :param pivot_col: Index of selected pivot column.
        :return: Index of pivot row with smallest non-negative ratio.
        """
        ratios = self.tableau[:-1, -1] / self.tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        return np.argmin(ratios)

    def _get_pivot_row_new(self, pivot_col):
        """
        Selects leaving variable row requiring strictly positive denominators (new method).

        :param pivot_col: Index of selected pivot column.
        :return: Index of pivot row with smallest positive ratio.
        :raises ValueError: If all elements in pivot column are non-positive.
        """
        column = self.tableau[:-1, pivot_col]
        positive_mask = column > 0
        if not np.any(positive_mask):
            raise ValueError("The solution is unlimited")
        ratios = np.full_like(column, np.inf, dtype=float)
        ratios[positive_mask] = self.tableau[:-1, -1][positive_mask] / column[positive_mask]
        return np.argmin(ratios)

    def _get_pivot_row(self, pivot_col):
        """
        Chooses pivot row selection method based on old_get_pilot flag.

        :param pivot_col: Index of selected pivot column.
        :return: Index of pivot row.
        :raises ValueError: For new method if unbounded solution detected.
        """
        return self._get_pivot_row_old(pivot_col) if self.old_get_pilot else self._get_pivot_row_new(pivot_col)

    def _pivot(self, pivot_row, pivot_col):
        """
        Performs pivot operation to update tableau and basis.

        :param pivot_row: Index of row to pivot on.
        :param pivot_col: Index of column to pivot on.
        """
        self._print_pretamble(pivot_col, pivot_row)
        self.tableau[pivot_row, :] /= self.tableau[pivot_row, pivot_col]
        for i in range(len(self.tableau)):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]
        self.basis[pivot_row] = pivot_col
        self._print_after_tamble()

    def solve(self):
        """
        Executes simplex algorithm iterations until optimal solution is found.

        :return: Tuple containing optimal value and solution vector.
        :raises ValueError: If problem is unbounded.
        """
        while np.any(self.tableau[-1, :-1] < 0):
            pivot_col = self._get_pivot_column()
            pivot_row = self._get_pivot_row(pivot_col)
            if np.all(self.tableau[:-1, pivot_col] <= 0):
                raise ValueError("The solution is unlimited")
            self._pivot(pivot_row, pivot_col)

        optimal_value = -self.tableau[-1, -1]
        solution = self._get_solution()
        return optimal_value, solution

    def _get_solution(self):
        """
        Extracts solution values from the optimized tableau.

        :return: Numpy array with values for original decision variables.
        """
        solution = np.zeros(self.num_variables)
        for i, var_index in enumerate(self.basis):
            if var_index < self.num_variables:
                solution[var_index] = self.tableau[i, -1]
        return solution

    def _print_pretamble(self, pivot_col, pivot_row):
        """
        Prints pre-pivot information including current tableau state.

        :param pivot_col: Selected pivot column index.
        :param pivot_row: Selected pivot row index.
        """
        if self._print:
            print(f"\nSelect the lead column: {pivot_col}, leading line: {pivot_row}")
            print("Current simplex table before calculation:")
            self._print_tableau()

    def _print_after_tamble(self):
        """Prints updated tableau state after pivoting operation."""
        if self._print:
            print("Onovlena simplex-table:")
            self._print_tableau()

    def _print_tableau(self):
        """Displays current tableau in formatted table using tabulate."""
        headers = [f"x{i}" for i in range(self.num_variables)] + \
                  [f"s{i}" for i in range(self.num_constraints)] + ["b"]

        row_labels = [f"B{self.basis[i]}" for i in range(self.num_constraints)] + ["Z"]
        table = np.round(self.tableau, 4).tolist()

        print(tabulate(table, headers=headers, showindex=row_labels, tablefmt="fancy_grid"))

    def _get_alternative_solution(self, pivot_col):
        """
        Generates alternative optimal solution through additional pivot.

        :param pivot_col: Column index for alternative pivot.
        :return: Alternative solution vector.
        :raises ValueError: If no valid pivot operation possible.
        """
        pivot_row = self._get_pivot_row(pivot_col)
        if np.all(self.tableau[:-1, pivot_col] <= 0):
            raise ValueError("No alternative solution can be found")

        tableau_copy = self.tableau.copy()
        basis_copy = self.basis.copy()

        self._pivot(pivot_row, pivot_col)
        alternative_solution = self._get_solution()

        self.tableau = tableau_copy
        self.basis = basis_copy
        return alternative_solution

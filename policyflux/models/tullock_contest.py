"""
Tullock Contest Model for rent-seeking and competitive expenditure.

The Tullock contest model represents a competition where actors invest
resources (expenditures) to win a prize. The probability of winning depends
on relative expenditures, with a contest parameter controlling the returns to scale.
"""

from typing import Any

from policyflux.exceptions import ValidationError


class TullockContest:
    """
    Tullock Contest Model for modeling competitive rent-seeking behavior.

    In a Tullock contest, n contestants compete for a prize by choosing
    expenditure levels e_i. Contestant i wins with probability:

        P_i = e_i^r / (sum_j e_j^r)

    where r is the contest parameter (0 < r <= 1).
    - r close to 1: more dispersed wins (small expenditures matter)
    - r close to 0: more concentrated wins (large expenditures dominate)
    """

    def __init__(self, n_contestants: int, prize_value: float = 1.0, r: float = 0.5) -> None:
        """
        Initialize Tullock Contest.

        Args:
            n_contestants: Number of contestants in the contest
            prize_value: Value of the prize being contested
            r: Contest parameter (0 < r <= 1) controlling elasticity of winning
        """
        if n_contestants < 2:
            raise ValidationError("Contest must have at least 2 contestants")
        if not 0.0 < r <= 1.0:
            raise ValidationError("Contest parameter r must be in (0, 1]")
        if prize_value <= 0:
            raise ValidationError("Prize value must be positive")

        self.n_contestants: int = n_contestants
        self.prize_value: float = prize_value
        self.r: float = r

        # Track expenditures and outcomes
        self.expenditures: list[float] = [0.0] * n_contestants
        self.win_probabilities: list[float] = [0.0] * n_contestants
        self.total_expenditure: float = 0.0

    def set_expenditure(self, contestant_id: int, expenditure: float) -> None:
        """
        Set the expenditure level for a contestant.

        Args:
            contestant_id: ID of the contestant (0 to n_contestants-1)
            expenditure: Expenditure level (non-negative)
        """
        if not 0 <= contestant_id < self.n_contestants:
            raise ValidationError(f"Contestant ID {contestant_id} out of range")
        if expenditure < 0:
            raise ValidationError("Expenditure cannot be negative")

        # Update total expenditure
        self.total_expenditure -= self.expenditures[contestant_id]
        self.total_expenditure += expenditure
        self.expenditures[contestant_id] = expenditure

    def get_expenditure(self, contestant_id: int) -> float:
        """Get the expenditure of a contestant."""
        if not 0 <= contestant_id < self.n_contestants:
            raise ValidationError(f"Contestant ID {contestant_id} out of range")
        return self.expenditures[contestant_id]

    def compute_win_probabilities(self) -> list[float]:
        """
        Compute winning probability for each contestant.

        Returns:
            List of probabilities summing to 1.0
        """
        if self.total_expenditure <= 0:
            # If no expenditure, equal probabilities
            self.win_probabilities = [1.0 / self.n_contestants] * self.n_contestants
        else:
            # Tullock CSF: P_i = e_i^r / sum(e_j^r)
            weighted_expenditures = [e**self.r for e in self.expenditures]
            total_weighted = sum(weighted_expenditures)

            if total_weighted > 0:
                self.win_probabilities = [e / total_weighted for e in weighted_expenditures]
            else:
                self.win_probabilities = [1.0 / self.n_contestants] * self.n_contestants

        return self.win_probabilities

    def get_win_probability(self, contestant_id: int) -> float:
        """Get the winning probability of a specific contestant."""
        if not 0 <= contestant_id < self.n_contestants:
            raise ValidationError(f"Contestant ID {contestant_id} out of range")
        if not self.win_probabilities:
            self.compute_win_probabilities()
        return self.win_probabilities[contestant_id]

    def get_expected_payoff(self, contestant_id: int) -> float:
        """
        Get the expected payoff for a contestant.

        Expected payoff = P_i * V - e_i
        where P_i is winning probability, V is prize value, e_i is expenditure.
        """
        if not 0 <= contestant_id < self.n_contestants:
            raise ValidationError(f"Contestant ID {contestant_id} out of range")

        win_prob = self.get_win_probability(contestant_id)
        expenditure = self.expenditures[contestant_id]
        return win_prob * self.prize_value - expenditure

    def get_all_payoffs(self) -> list[float]:
        """Get expected payoffs for all contestants."""
        return [self.get_expected_payoff(i) for i in range(self.n_contestants)]

    def get_total_waste(self) -> float:
        """
        Get total expenditure in the contest.

        This represents rent dissipation - resources spent on the contest
        but not creating value.
        """
        return self.total_expenditure

    def get_efficiency(self) -> float:
        """
        Get efficiency of the contest.

        Efficiency = (Prize value - Total expenditure) / Prize value
        Low efficiency means much is wasted on the contest.
        """
        if self.prize_value <= 0:
            return 0.0
        return max(0.0, (self.prize_value - self.total_expenditure) / self.prize_value)

    def get_rent_dissipation_ratio(self) -> float:
        """
        Get the ratio of total expenditure to prize value.

        Higher ratio means more rent dissipation.
        """
        if self.prize_value <= 0:
            return 0.0
        return self.total_expenditure / self.prize_value

    def simulate_equilibrium(
        self,
        initial_expenditures: list[float] | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """
        Simulate convergence to Nash equilibrium.

        Uses best-response dynamics where each contestant adjusts
        expenditure based on others' expenditures.

        Args:
            initial_expenditures: Starting expenditure levels
            max_iterations: Maximum iterations for simulation
            tolerance: Convergence tolerance

        Returns:
            Dictionary with equilibrium expenditures and statistics
        """
        if initial_expenditures is None:
            initial_expenditures = [1.0 / self.n_contestants] * self.n_contestants

        if len(initial_expenditures) != self.n_contestants:
            raise ValidationError("Initial expenditures must match number of contestants")

        # Initialize
        for i, exp in enumerate(initial_expenditures):
            self.set_expenditure(i, exp)

        convergence_history = []

        for iteration in range(max_iterations):
            old_expenditures = self.expenditures.copy()

            # Best response for each contestant
            for i in range(self.n_contestants):
                # Simplified BR: respond to others' total
                others_total = self.total_expenditure - self.expenditures[i]

                # Optimal expenditure increases with others' expenditure
                # BR heuristic: best response is proportional to others' expenditure
                if others_total > 0:
                    new_exp = others_total / (2.0 * self.n_contestants)
                else:
                    new_exp = 1.0 / self.n_contestants

                self.set_expenditure(i, new_exp)

            # Check convergence
            max_change = max(
                abs(old_expenditures[i] - self.expenditures[i]) for i in range(self.n_contestants)
            )
            convergence_history.append(max_change)

            if max_change < tolerance:
                break

        self.compute_win_probabilities()

        return {
            "equilibrium_expenditures": self.expenditures.copy(),
            "win_probabilities": self.win_probabilities.copy(),
            "expected_payoffs": self.get_all_payoffs(),
            "total_expenditure": self.total_expenditure,
            "efficiency": self.get_efficiency(),
            "rent_dissipation_ratio": self.get_rent_dissipation_ratio(),
            "iterations": len(convergence_history),
            "converged": max_change < tolerance if convergence_history else False,
            "convergence_history": convergence_history,
        }

    def analyze_contest_parameter_sensitivity(
        self, r_values: list[float] | None = None
    ) -> dict[str, Any]:
        """
        Analyze how contest parameter r affects outcomes.

        Args:
            r_values: List of r values to test (default: [0.1, 0.3, 0.5, 0.7, 0.9])

        Returns:
            Dictionary with sensitivity analysis results
        """
        if r_values is None:
            r_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        results = {}

        for r_test in r_values:
            if not 0.0 < r_test <= 1.0:
                continue

            # Create contest with this r
            contest = TullockContest(self.n_contestants, self.prize_value, r_test)
            for i in range(self.n_contestants):
                contest.set_expenditure(i, self.expenditures[i])

            contest.compute_win_probabilities()

            results[r_test] = {
                "win_probabilities": contest.win_probabilities.copy(),
                "hhi": self._compute_hhi(contest.win_probabilities),  # Concentration index
                "efficiency": contest.get_efficiency(),
            }

        return results

    @staticmethod
    def _compute_hhi(probabilities: list[float]) -> float:
        """
        Compute Herfindahl-Hirschman Index (HHI) for concentration.

        HHI = sum(p_i^2). Ranges from 1/n to 1.
        Closer to 1 means more concentrated.
        """
        return sum(p**2 for p in probabilities)

    def reset(self) -> None:
        """Reset the contest to initial state."""
        self.expenditures = [0.0] * self.n_contestants
        self.win_probabilities = [0.0] * self.n_contestants
        self.total_expenditure = 0.0

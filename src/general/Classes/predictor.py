from src.general.Classes.solution import Solution

class Predictor:
    def __init__(
            self,
    ):
        self.name: str

    def predict(self, solution: Solution):
        objective = 0
        solution.set_simulated_objective(objective)

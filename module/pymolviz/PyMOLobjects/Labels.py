import numpy as np
from ..Displayable import Displayable


class Labels(Displayable):
    def __init__(self, positions, labels, name = None, state = 1, **kwargs):
        """ Represents a set of labels.

        Args:
            positions (np.array): The positions of the labels.
            labels (list): The labels.
            name (str): Optional. The name of the data. Defaults to None.
            state (int): Optional. The state of the data. Defaults to 1.

        """
        positions = np.array(positions)
        if len(positions.shape) == 1:
            positions = np.array([positions])
            labels = [labels]
        self.positions = positions
        self.labels = labels
        self.state = state

        super().__init__(name = name, **kwargs)

    def _script_string(self):
        result = []
        for label, position in zip(self.labels, self.positions):
            result.append(f"""cmd.pseudoatom("{self.name}", label="{label}", pos = {position.tolist()}, state = {self.state})""")
        return "\n".join(result)

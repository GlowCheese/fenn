from pathlib import Path
from typing import List, Optional, Union

from fenn.logging import Logger


class Checkpoint:
    """Checkpoint training state at the given epochs."""

    def __init__(
        self,
        *,
        name: str = "checkpoint",
        dir: Union[Path, str],
        epochs: Optional[Union[int, List[int]]] = None,
        save_best: bool = True,
    ):
        """Initialize the checkpoint configuration.

        Args:
            name: The name of the checkpoint file.
            dir: The directory to save checkpoints to.
            epochs: The epochs at which to save checkpoints.
            save_best: Whether to checkpoint the best model (based on validation or training loss).
        """

        self._logger = Logger()

        self.name = name
        self.dir = Path(dir)
        self.epochs = epochs
        self.save_best = save_best

    def _setup(self):
        """Set up the checkpoint directory and checks."""
        self.dir.mkdir(parents=True, exist_ok=True)

        if self.epochs is None and not self.save_best:
            self._logger.system_warning(
                "Checkpoint configuration is passed, but both `epochs` and `save_best` are unset.\n"
                "Models will not be checkpointed."
            )
            return

        if self.epochs is not None:
            self._logger.system_info(
                f"Checkpointing enabled. Checkpoints will be saved to {self.dir} every {self.epochs} epochs."
            )

        if self.save_best:
            self._logger.system_info(
                f"Best model checkpointing enabled. Best model will be saved to {self.dir}."
            )

        return self

import builtins
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from colorama import Fore, Style

class baseLogger(ABC):
    @abstractmethod
    def system_info(self, message: str) -> None:
        pass

    @abstractmethod
    def system_warning(self, message: str) -> None:
        pass

    @abstractmethod
    def system_exception(self, message: str) -> None:
        pass

    @abstractmethod
    def user_info(self, message: str) -> None:
        pass

    @abstractmethod
    def user_warning(self, message: str) -> None:
        pass

    @abstractmethod
    def user_exception(self, message: str) -> None:
        pass
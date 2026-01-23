from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class VerificationResult:
    status: str  # "Verified", "Failed", "Unknown", "Error"
    output: str
    details: Dict[str, Any] = None

class BaseVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        code: str,
        loop_invariants: Optional[Dict[int, List[str]]] = None,
        loop_rankings: Optional[Dict[int, str]] = None,
    ) -> VerificationResult:
        """
        Verify the given properties on the code.
        """
        pass

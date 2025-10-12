# Literal Enum:
#  - no additional values can be added over time.
#  - Sent as-is, not in hex symbol
from enum import auto
from typing import List

from gw.enums import GwStrEnum


class RepresentationStatus(GwStrEnum):

    Unknown = auto()
    ListeningToAtn = auto()
    NotListeningToAtn = auto()

    @classmethod
    def values(cls) -> List[str]:
        """
        Returns enum choices
        """
        return [elt.value for elt in cls]

    @classmethod
    def default(cls) -> "RepresentationStatus":
        return cls.Unknown

    @classmethod
    def enum_name(cls) -> str:
        return "gw0.representation.status"

    @classmethod
    def version(cls) -> str:
        return "000"

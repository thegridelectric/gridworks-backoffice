"""
ASL Codec - Self-contained serialization system
Generated for: gridworks-backoffice
Generated on: 2025-10-07
Selected types: 12
"""
import re
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
from collections import defaultdict
from pydantic import BaseModel, ConfigDict, ValidationError


logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

snake_add_underscore_to_camel_pattern = re.compile(r"(?<!^)(?=[A-Z])")


def is_pascal_case(s: str) -> bool:
    return re.match(r"^[A-Z][a-zA-Z0-9]*$", s) is not None


def recursively_pascal(d: dict) -> bool:
    """
    Checks that all dict keys are pascal case, all the way down
    """
    if isinstance(d, dict):
        # Check if all keys in the dictionary are in PascalCase
        for key in d:
            if not is_pascal_case(key):
                return False
            # Recursively check nested dictionaries or lists
            if not recursively_pascal(d[key]):
                return False
    elif isinstance(d, list):
        # Recursively check if dictionaries or lists inside a list pass the test
        for item in d:
            if not recursively_pascal(item):
                return False
    # If it's neither a dict nor a list, return True (nothing to check)
    return True


def pascal_to_snake(name: str) -> str:
    return snake_add_underscore_to_camel_pattern.sub("_", name).lower()


def snake_to_pascal(word: str) -> str:
    return "".join(x.capitalize() or "_" for x in word.split("_"))


# ============================================================================
# BASE TYPE CLASS
# ============================================================================

class AslError(Exception):
    """Base exception for ASL-related errors."""
    pass


T = TypeVar("T", bound="AslType")

class AslType(BaseModel):
    """
    Base class for the Application Shared Language Types

    Notes:
        - `type_name`: Must follow left-right-dot (LRD) format. Subclasses
        are expected to overwrite this with a literal. The format is enforced
        by the ASL Type Registry , which is the source of truth
        - `version`: Must be  a three-digit string (e.g. "000", "001"), or None.
        Subclasses are expected to overwrite this with either a literal or a
        string, with the literal (strict versioning) being the default. The
        format is enforced by the ASL Type Registry, which is the source of truth.

    For more information:
      - [GridWorks ASL Docs](https://gridworks-asl.readthedocs.io)
    """
    
    type_name: str
    version: str | None = None
    
    model_config = ConfigDict(
        alias_generator=snake_to_pascal,
        frozen=True,
        populate_by_name=True,
        extra="forbid",
    )

    def to_bytes(self) -> bytes:
        return self.model_dump_json(exclude_none=True, by_alias=True).encode()

    def to_dict(self) -> Dict[str, Any]:
        bytes = self.model_dump_json(exclude_none=True, by_alias=True)
        return json.loads(bytes)

    @classmethod
    def from_bytes(cls, bytes) -> T:
        try:
            d = json.loads(bytes)
        except TypeError as e:
            raise AslError("Type must be string or bytes!") from e
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls: Type[T], d: dict) -> T:
        if not recursively_pascal(d):
            raise AslError(
                f"Dictionary keys must be recursively PascalCase. "
                f"Found: {d}. Consider checking nested structures."
            )
        try:
            t = cls.model_validate(d)
        except ValidationError as e:
            raise AslError(f"Validation failed for {cls.__name__}: {e}") from e
        return t

    @classmethod
    def get_schema_info(cls) -> Dict[str, Any]:
        """Return schema information for this type."""
        return {
            "type_name": cls.type_name_value(),
            "version": cls.version_value(),
            "fields": list(cls.model_fields.keys()),
        }

    def __repr__(self) -> str:
        """Provide clear representation for debugging and logging."""
        return f"{self.__class__.__name__}(type_name='{self.type_name}', version='{self.version}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.type_name_value()}.{self.version_value()}"

    @classmethod
    def type_name_value(cls) -> str:
        # Automatically return the type_name defined in the subclass
        return cls.model_fields["type_name"].default

    @classmethod
    def version_value(cls) -> str | None:
        # return the Version defined in the subclass
        return cls.model_fields["version"].default


# ============================================================================
# CODEC CLASS
# ============================================================================

class AslCodec:
    """
    Codec for this repository's ASL types.
    Handles version flexibility and naming convention conversion.
    """
    
    def __init__(self):
        """
        Initialize the codec with auto-discovered types
        """
        self.registry = get_current_types()
        self.old_versions = get_old_versions()

        # Validate that all old versions have a current target
        for type_name in self.old_versions:
            if type_name not in self.registry:
                raise ValueError(
                    f"Old versions found for '{type_name}' but no current version exists. "
                    f"Old versions: {list(self.old_versions[type_name].keys())}"
                )

    def from_dict(self, data: dict) -> AslType:
        """Decode a dictionary to the appropriate AslType."""
        type_name = data.get("TypeName")
        if not type_name:
            raise ValueError("Missing TypeName field")
        
        version = data.get("Version")
        
        if type_name not in self.registry:
            raise ValueError(
                f"Unknown type: {type_name}. "
                f"Known types: {self.registry.keys()}"
            )
        
        current_cls = self.registry[type_name]
        current_version = current_cls.version_value()

        # Fast path: version matches current
        if version == current_version:
            return current_cls.from_dict(data)
        
        # Translation path: we have an old version
        if type_name in self.old_versions and version in self.old_versions[type_name]:
            logger.warning(
                f"Translating {type_name} from v{version} to v{current_version}"
            )
            old_cls = self.old_versions[type_name][version]
            old_instance = old_cls.from_dict(data)
            return old_instance.to_latest()

        # Fallback: try to decode with current version anyway
        logger.warning(
            f"Unknown version {version} for {type_name}, "
            f"attempting decode with current v{current_version}"
        )

        data = dict(data)  # Make a copy
        data["Version"] = current_version
        try:
            return current_cls.from_dict(data)
        except (ValidationError, AslError) as e:
            logger.warning(f"Stripping unknown fields and retrying")
            data = self._strip_unknown_fields(data, current_cls)
            return current_cls.from_dict(data)

    
    def from_bytes(self, data: bytes) -> AslType:
        """Decode JSON bytes to the appropriate AslType"""
        try:
            d = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
        return self.from_dict(d)
    
    def to_bytes(self, msg: AslType) -> bytes:
        """Encode an AslType to JSON bytes"""
        return msg.to_bytes()

    def _strip_unknown_fields(self, data: dict, cls: Type[AslType]) -> dict:
        """Remove fields not recognized by the target class."""
        valid_fields = set()
        for field_name, field_info in cls.model_fields.items():
            valid_fields.add(field_name)
            valid_fields.add(snake_to_pascal(field_name))
            if field_info.alias:
                valid_fields.add(field_info.alias)
        
        cleaned = {}
        for key, value in data.items():
            if key in valid_fields or pascal_to_snake(key) in valid_fields:
                cleaned[key] = value
        
        return cleaned


# ============================================================================
# AUTO-DISCOVERY OF TYPES IN THIS REPO
# ============================================================================

def get_current_types() -> Dict[str, Type[AslType]]:
    """
    Returns the types declared in `asl/types/__init__.py`
    """
    from gbo.asl import types
    registry = {}
    for name in types.__all__:
        cls = getattr(types, name)
        type_name = cls.type_name_value()
        registry[type_name] = cls
    
    return registry

def get_old_versions() -> Dict[str, Dict[Optional[str], Type[AslType]]]:
    """
     Returns a registry of old versions organized by type_name and version.
    Structure: {type_name: {version: class}}
    """
    from gbo.asl.types import old_versions
    old_types = [getattr(old_versions, name) for name in old_versions.__all__]

    old_registry: Dict[str, Dict[Optional[str], Type[AslType]]] = defaultdict(dict)

    for cls in old_types:
        type_name = cls.type_name_value()
        version = cls.version_value() # Could be None
        
        if type_name in old_registry and version in old_registry[type_name]:
            existing = old_registry[type_name][version]
            if existing != cls:
                raise ValueError(
                    f"Duplicate registration: {type_name} v{version} "
                    f"already registered to {existing.__name__}"
                )

        old_registry[type_name][version] = cls

    return old_registry

# ============================================================================
# DEFAULT CODEC INSTANCE FOR THIS REPOSITORY
# ============================================================================

# Create a default codec instance that can be imported
default_codec = AslCodec()
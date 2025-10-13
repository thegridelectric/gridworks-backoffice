# ASL (Application Shared Language)

This directory contains the type definitions and serialization codec for the GridWorks Backoffice application.

## What is ASL?

ASL (Application Shared Language) is a protocol framework similar to Protocol Buffers or OpenAPI, but designed for distributed peer-to-peer systems rather than client-server architectures.

**Key concepts:**
- **À la carte selection** - Choose only the types you need (unlike monolithic SDKs)
- **Self-contained generation** - Get a complete `asl/` directory with no external dependencies
- **Version independence** - Each repo can use different versions without conflicts
- **Language neutral** - JSON Schema specs that generate idiomatic Python (or any language)

ASL provides structured, validated Python types for data that needs to be:
- Stored in the database
- Sent between frontend and backend  
- Shared with other GridWorks services or external systems

Think of it as the evolution from REST APIs (where the server dictates the contract) to shared vocabulary (where peers collaborate on equal terms).

## Benefits Over JSON Blobs

### Type Safety
```python
# Current: JSON blob - no validation
house.address = {
    "street": "123 Main",
    "latitude": 999,  # Invalid but stored anyway
}

# With ASL: Validated types
address = HouseAddress(
    street="123 Main",
    latitude_micro_deg=999_000_000,  # Raises error: out of Earth bounds
)
```

### Autocomplete & Documentation
```python
# Current: Must remember field names
address_dict["stret"]  # Typo - silent failure

# With ASL: IDE helps you
address.street  # Autocomplete works
address.lat  # Convenient property (converts from micro degrees)
```

### Consistent Serialization
```python
# ASL handles PascalCase for wire format automatically
address.to_dict()  # {"Street": "123 Main", "LatitudeMicroDeg": 42358430, ...}
address.to_bytes()  # Ready for network transmission
```

## Migration Example

### Current Code
```python
# In models.py
class House(Base):
    address = Column(JSON)  # Unstructured

# In routes
house.address = request.json["address"]  # No validation

# In templates
if house.address.get("street"):  # Hope the field exists
```

### With ASL Types
```python
# In models.py - still store as JSON but validated
class House(Base):
    address = Column(JSON)
    
    @property
    def address_obj(self) -> HouseAddress:
        """Get typed address object."""
        return HouseAddress.from_dict(self.address)
    
    def set_address(self, address: HouseAddress):
        """Set address from typed object."""
        self.address = address.to_dict()

# In routes
address = HouseAddress(
    street=request.json["street"],
    city=request.json["city"],
    # ... validates automatically
)
house.set_address(address)

# In templates - same JSON structure works
{{ house.address.Street }}  # Still works
```

## Frontend/Backend Separation

ASL types define the contract between frontend and backend:

```python
# Backend API endpoint
@app.post("/api/house/{house_id}/address")
def update_address(house_id: int, address_data: dict):
    # Validate incoming data
    address = HouseAddress.from_dict(address_data)
    
    # Use in business logic with full type safety
    if address.lat > 45.0:  # Property gives float from int storage
        # Special handling for northern locations
    
    # Store validated data
    house.address = address.to_dict()
```

The frontend sends/receives the same JSON structure, but the backend has type safety and validation.

## Using the Codec

```python
from gbo.asl import AslCodec, HouseAddress

codec = AslCodec()

# Decode incoming message
address = codec.from_dict(request.json)

# Handle version migrations automatically
old_data = {"TypeName": "gw0.house.address", "Version": "000", ...}
address = codec.from_dict(old_data)  # Automatically migrates to current version
```

## Next Steps

1. Start using typed properties for new features
2. Gradually migrate existing JSON fields to use validation
3. Consider separate API service using these same types

## Structure

- `codec.py` - Base type class and serialization codec
- `enums/` - Enumerated types (RepresentationStatus, etc.)
- `types/` - Message type definitions
- `types/old_versions/` - Previous versions for migration
- `property_format.py` - Field validators
- `tests/` - Basic functionality tests

## Source: GridWorks ASL Registry

These types were generated from the [GridWorks ASL Registry](https://github.com/thegridelectric/gridworks-asl), which maintains the complete vocabulary for GridWorks systems. The registry works as an "à la carte menu" - you select only the types you need, and the generator creates a self-contained `asl/` directory for your repository.

- **Browse available types**: [electricity.works](https://asl.electricity.works) COMING SOON
- **Validate messages**: [api.electricity.works](https://api.electricity.works) COMING SOON
- **Add new types**: Submit PRs to [gridworks-asl](https://github.com/thegridelectric/gridworks-asl)

To regenerate this directory with updated or additional types:
```bash
# Coming soon
gridworks-asl generate --types gw0.house.address,gw0.house.contact,gw0.house.status
```

Unlike traditional package dependencies, each generated `asl/` directory is completely self-contained - no imports from gridworks-asl needed.

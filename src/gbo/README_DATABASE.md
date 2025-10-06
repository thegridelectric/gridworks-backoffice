# Database Setup for GridWorks Backoffice

This directory contains the database configuration and migration setup for the GridWorks Backoffice application using PostgreSQL and Alembic.

## Prerequisites

1. PostgreSQL server running (with psql accessible)
2. Environment variable `GBO_DB_URL` set to your PostgreSQL connection string
3. Python dependencies installed via `uv`

## Setup Instructions

### 1. Set Environment Variable

Set the `GBO_DB_URL` environment variable to your PostgreSQL connection string:

```bash
export GBO_DB_URL='postgresql://postgres:password@localhost/backofficedb'
```

Replace `password` with your actual PostgreSQL password and adjust the host/port as needed.

### 2. Install Dependencies

From the project root:

```bash
uv pip install -r requirements.txt
```

### 3. Initialize Database

Run the setup script to create the database:

```bash
cd src/gbo
uv run python setup_database.py
```

### 4. Create Initial Migration

Generate the initial migration based on your models:

```bash
cd src/gbo
uv run alembic revision --autogenerate -m "Initial migration"
```

### 5. Apply Migration

Apply the migration to create the database schema:

```bash
cd src/gbo
uv run alembic upgrade head
```

## File Structure

- `database.py` - Database configuration and connection setup
- `models.py` - SQLAlchemy models for the application
- `alembic/` - Alembic migration files and configuration
- `alembic.ini` - Alembic configuration file
- `setup_database.py` - Database setup script

## Database Models

The following models are defined:

- `House` - House/household data with address, contacts, and hardware info
- `User` - User authentication and management
- `HourlyElectricity` - Hourly electricity consumption data
- `EnergyData` - General energy data storage
- `SystemLog` - System logging and audit trails

## Working with Migrations

### Creating New Migrations

1. Modify your models in `models.py`
2. Generate a new migration:
   ```bash
   cd src/gbo
   uv run alembic revision --autogenerate -m "Description of changes"
   ```
3. Apply the migration:
   ```bash
   cd src/gbo
   uv run alembic upgrade head
   ```

### Migration Commands

- `alembic current` - Show current migration version
- `alembic history` - Show migration history
- `alembic upgrade head` - Apply all pending migrations
- `alembic downgrade -1` - Rollback one migration
- `alembic downgrade base` - Rollback all migrations

## Environment Variables

- `GBO_DB_URL` - PostgreSQL connection string (required)

Example: `postgresql://username:password@host:port/database_name`

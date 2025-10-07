# BackofficeDb Setup

This directory contains the database configuration and migration setup for the BackofficeDb database using PostgreSQL and Alembic.

## EC2 Instance Setup

### 1. Create EC2 Instance
- Create a new EC2 instance from the `journaldb-template` image
- Eventually add it to Route 53 for DNS resolution

### 2. Verify PostgreSQL Service
SSH onto the new instance and check that PostgreSQL is running:

```bash
sudo systemctl status postgresql
```

### 3. Create PostgreSQL Cluster
Create a new cluster using the default location:

```bash
sudo pg_createcluster 14 main --start
```

### 4. Configure PostgreSQL

**Edit PostgreSQL configuration:**
```bash
sudo nano /etc/postgresql/*/main/postgresql.conf
```
Add the following line:
```
listen_addresses = '*'
```

**Edit authentication configuration:**
```bash
sudo nano /etc/postgresql/*/main/pg_hba.conf
```
Add this line for remote connections:
```
host    all             all             0.0.0.0/0               md5
```
Change the local authentication method from "peer" to "md5":
```
local   all             all                                     md5
```

### 5. Restart PostgreSQL
Apply the configuration changes:

```bash
sudo systemctl restart postgresql
```

### 6. Add backoffice user and database
```bash
sudo -i -u postgres
psql
...
```

## Local setup Instructions

### 1. Set environment variable

Set the `GBO_DB_URL` environment variable in the `.env` file:

```
GBO_DB_URL = "postgresql://backofficedb:password@backofficedb.electricity.works/backofficedb"
```

### 2. Install dependencies

From the project root:

```bash
uv pip install -r requirements.txt
```

### 3. Initialize database

Run the setup script to create the database:

```bash
cd src/gbo
uv run python setup_database.py
```

### 4. Create initial migration

Generate the initial migration based on the models:

```bash
cd src/gbo
uv run alembic revision --autogenerate -m "Initial migration"
```

### 5. Apply migration

Apply the migration to create the database schema:

```bash
cd src/gbo
uv run alembic upgrade head
```

## File structure

- `database.py` - Database configuration and connection setup
- `models.py` - SQLAlchemy models for the application
- `alembic/` - Alembic migration files and configuration
- `alembic.ini` - Alembic configuration file
- `setup_database.py` - Database setup script

## Creating new migrations

1. Modify the models in `models.py`
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
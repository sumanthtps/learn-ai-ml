---
id: async-database
title: "15 · Async Database — PostgreSQL + SQLAlchemy + Alembic"
sidebar_label: "15 · Async Database"
sidebar_position: 15
tags: [postgresql, sqlalchemy, alembic, async, database, orm, migrations, intermediate]
---

# Async Database — PostgreSQL + SQLAlchemy + Alembic

> **Advanced Topic** — Replacing the JSON file with a real production-grade async PostgreSQL database.

---

## Visual Reference

![PostgreSQL elephant logo](https://commons.wikimedia.org/wiki/Special:Redirect/file/Postgresql_elephant.svg)

Source: [Wikimedia Commons - Postgresql elephant](https://commons.wikimedia.org/wiki/File:Postgresql_elephant.svg)

## Why We Need a Real Database

Throughout the CampusX series, we stored data in a JSON file. This was intentional — it let us focus on learning FastAPI without database complexity. But in production, a JSON file fails in three critical ways:

**Problem 1: Race Conditions (Concurrent Writes Corrupt Data)**

```
Request A reads:  {"P001": {...}, "P002": {...}}
Request B reads:  {"P001": {...}, "P002": {...}}
Request A adds P003 and writes the file
Request B adds P003 and writes the file  ← P003 from Request A is LOST
```

Two requests writing simultaneously can overwrite each other. With 10 users, this happens constantly.

**Problem 2: No Transactions**

```python
# What if the code crashes between these two operations?
db["P001"]["is_active"] = False       # patient deactivated
del db["all_patients"]["P001"]        # this line crashes due to a bug
# Now P001 is deactivated but still in the list — inconsistent state!
```

Databases guarantee **atomicity** — either all operations succeed together or none do.

**Problem 3: Performance**

Reading a 100,000-patient JSON file on every request takes 500ms. A PostgreSQL query with proper indexes takes 1ms.

---

## What is an ORM?

ORM stands for **Object-Relational Mapper**. It lets you work with your database using Python classes and objects instead of writing raw SQL queries.

Without ORM (raw SQL):
```python
cursor.execute("SELECT * FROM patients WHERE city = %s AND age > %s", ("Mumbai", 30))
rows = cursor.fetchall()
patients = [{"id": row[0], "name": row[1], ...} for row in rows]
```

With SQLAlchemy ORM:
```python
patients = await db.execute(
    select(Patient).where(Patient.city == "Mumbai", Patient.age > 30)
)
```

The ORM translates your Python code into SQL, handles connection pooling, prevents SQL injection, and maps database rows to Python objects automatically.

---

## Why Async Database Access?

When your FastAPI endpoint queries the database, your code is waiting for the database server to respond. This is I/O — the CPU is idle, just waiting.

```
Synchronous (blocking):
Endpoint called → query sent → CPU WAITS (100ms) → result → response
                               ↑ 100ms wasted, no other requests can run

Asynchronous (non-blocking):
Endpoint called → query sent → CPU handles other requests → result arrives → response
                               ↑ CPU stays busy, serving other requests simultaneously
```

With `asyncpg` (an async PostgreSQL driver) + async SQLAlchemy, your FastAPI event loop can handle hundreds of simultaneous database queries without blocking.

---

## Installation

```bash
# Core packages
pip install sqlalchemy[asyncio]   # SQLAlchemy with async support
pip install asyncpg               # async PostgreSQL driver
pip install alembic               # database migration tool
pip install pydantic-settings     # settings management

# For development: easy PostgreSQL setup
# Install PostgreSQL locally or use Docker:
docker run -d \
  --name postgres-dev \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mlapi \
  -p 5432:5432 \
  postgres:16-alpine
```

---

## Project Structure

```
app/
├── main.py                    ← FastAPI app + lifespan
├── core/
│   ├── config.py              ← Settings from environment
│   └── database.py            ← Engine + session factory + get_db
├── models/
│   ├── base.py                ← DeclarativeBase + TimestampMixin
│   └── patient.py             ← Patient ORM model (maps to DB table)
├── schemas/
│   └── patient.py             ← Pydantic models for API input/output
├── repositories/
│   └── patient.py             ← Database access layer (queries)
├── routers/
│   └── patients.py            ← HTTP endpoints
└── alembic/
    ├── env.py                 ← Alembic configuration
    └── versions/              ← Migration files (one per schema change)
```

---

## Step 1: Configuration

```python title="app/core/config.py"
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """
    All configuration loaded from environment variables.
    
    In development: create a .env file with these values.
    In production: set real environment variables on your server.
    NEVER hardcode database passwords or secrets in Python files!
    """
    
    # Database connection
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "mlapi"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    
    # Connection pool settings
    # pool_size: number of persistent connections to maintain
    # max_overflow: additional connections allowed when pool is full
    db_pool_size: int = 10
    db_max_overflow: int = 5
    
    # App settings
    model_path: str = "artifacts/model.pkl"
    log_level: str = "INFO"
    
    @property
    def async_database_url(self) -> str:
        """
        URL for async connections using asyncpg driver.
        postgresql+asyncpg:// tells SQLAlchemy to use asyncpg.
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def sync_database_url(self) -> str:
        """
        URL for synchronous connections using psycopg2.
        Alembic (migration tool) needs a sync connection.
        """
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

---

## Step 2: Database Engine and Session

```python title="app/core/database.py"
from sqlalchemy.ext.asyncio import (
    create_async_engine,      # async version of create_engine
    AsyncSession,             # async version of Session
    async_sessionmaker,       # factory for creating sessions
)
from core.config import get_settings

settings = get_settings()

# ─── The Engine ──────────────────────────────────────────────────
# The engine is the connection to your database.
# It manages a pool of actual database connections.
# pool_pre_ping=True: test each connection before using it (handles stale connections)
engine = create_async_engine(
    settings.async_database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,    # test connections before using from pool
    echo=False,            # set True to print all SQL queries (debugging only!)
)

# ─── Session Factory ──────────────────────────────────────────────
# A session represents one "unit of work" with the database.
# Think of it like a shopping cart: you add/update/delete items,
# then commit (checkout) to make all changes permanent.
# 
# expire_on_commit=False: don't expire attributes after commit.
# Without this, accessing obj.name after commit triggers a new DB query.
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ─── Dependency: Per-Request Session ──────────────────────────────
async def get_db():
    """
    FastAPI dependency that provides a fresh database session per request.
    
    The 'async with' context manager:
    - Opens a session at the start of the request
    - Yields it to the endpoint function
    - Commits on success OR rolls back on error
    - Always closes the session (releases connection back to pool)
    
    Usage in endpoints:
        async def my_endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()   # commit if no exception
        except Exception:
            await session.rollback() # undo all changes if anything went wrong
            raise
        # session is automatically closed after exiting 'async with'
```

---

## Step 3: ORM Models

ORM models define your database tables. Each Python class = one database table. Each class attribute = one database column.

```python title="app/models/base.py"
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy import DateTime, func
from datetime import datetime

class Base(DeclarativeBase):
    """
    The base class all ORM models inherit from.
    DeclarativeBase is the modern SQLAlchemy 2.0 way to define models.
    """
    pass

class TimestampMixin:
    """
    A mixin (shared code) that adds created_at and updated_at timestamps.
    Any model that inherits this gets these columns automatically.
    
    server_default=func.now(): PostgreSQL fills this automatically on INSERT.
    onupdate=func.now(): PostgreSQL updates this automatically on UPDATE.
    
    This means you never need to manually set these fields in your code.
    """
    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
```

```python title="app/models/patient.py"
from sqlalchemy import String, Integer, Float, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, TimestampMixin

class Patient(Base, TimestampMixin):
    """
    Represents the 'patients' table in PostgreSQL.
    
    Mapped[str] is a type annotation that tells SQLAlchemy (and your IDE)
    what Python type this column holds. SQLAlchemy uses this to:
    1. Choose the right SQL column type
    2. Provide type hints in your editor
    3. Validate assignments
    """
    __tablename__ = "patients"    # actual PostgreSQL table name
    
    # Primary key: auto-incrementing integer ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # External ID (the P001 style ID we show to API clients)
    # unique=True: PostgreSQL enforces uniqueness
    # index=True: creates a B-tree index for fast lookups by external_id
    external_id: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    smoker: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Soft delete flag — we never actually delete rows
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Composite index for commonly combined filters
    # This makes queries like "WHERE city = 'Mumbai' AND smoker = true" fast
    __table_args__ = (
        Index("ix_patients_city_smoker", "city", "smoker"),
    )
    
    @property
    def bmi(self) -> float:
        """Calculated from weight and height, not stored in DB."""
        return round(self.weight / ((self.height / 100) ** 2), 2)
```

---

## Step 4: Pydantic Schemas for the API

Separate from ORM models, these define what the API accepts and returns:

```python title="app/schemas/patient.py"
from pydantic import BaseModel, Field, computed_field
from typing import Optional
from datetime import datetime

class PatientCreate(BaseModel):
    """What the client sends when creating a patient."""
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=150)
    city: str = Field(min_length=2)
    weight: float = Field(gt=0, description="Weight in kilograms")
    height: float = Field(gt=0, description="Height in centimeters")
    smoker: bool = False

class PatientUpdate(BaseModel):
    """What the client sends when partially updating a patient (PATCH)."""
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    smoker: Optional[bool] = None

class PatientResponse(BaseModel):
    """What the API returns. Includes server-generated fields like id and bmi."""
    id: int
    external_id: str
    name: str
    age: int
    city: str
    weight: float
    height: float
    smoker: bool
    bmi: float
    created_at: datetime
    
    # from_attributes=True allows creating this from an ORM model instance
    # without this, Pydantic can't read SQLAlchemy ORM objects
    model_config = {"from_attributes": True}

class PaginatedPatients(BaseModel):
    """Paginated list response."""
    total: int
    page: int
    page_size: int
    total_pages: int
    data: list[PatientResponse]
```

---

## Step 5: Repository Pattern — Clean Database Access

The **Repository Pattern** is a design principle: separate your database access code from your endpoint logic. Benefits:
- Endpoints stay clean and focused on HTTP concerns
- Database queries are easy to test in isolation
- Switching databases later only requires changing the repository

```python title="app/repositories/patient.py"
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload
from typing import Optional
from models.patient import Patient
import uuid

class PatientRepository:
    """
    All database operations for Patient records.
    
    The session is injected (passed in) rather than created here.
    This follows Dependency Injection — the repository doesn't
    know how the session was created, making it more flexible and testable.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, patient_id: int) -> Optional[Patient]:
        """Look up a patient by internal database ID."""
        # session.get() uses the identity map (in-memory cache) first
        # Only hits the database if not already loaded this session
        return await self.session.get(Patient, patient_id)
    
    async def get_by_external_id(self, external_id: str) -> Optional[Patient]:
        """
        Look up by the P001-style external ID that clients use.
        
        select(Patient): SELECT * FROM patients
        .where(...)    : WHERE external_id = :external_id AND is_deleted = false
        .one_or_none() : return the single result, or None if not found
        """
        result = await self.session.execute(
            select(Patient).where(
                Patient.external_id == external_id,
                Patient.is_deleted == False
            )
        )
        return result.scalar_one_or_none()
    
    async def search(
        self,
        city: Optional[str] = None,
        smoker: Optional[bool] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[Patient], int]:
        """
        Search patients with optional filters, returns (results, total_count).
        
        We build the query dynamically — conditions are only added if
        the corresponding filter was provided. This avoids messy if-else chains.
        """
        # Start with base query (only non-deleted patients)
        base_query = select(Patient).where(Patient.is_deleted == False)
        
        # Add optional filters
        if city:
            # ilike = case-insensitive LIKE
            # %city% = contains city anywhere in the string
            base_query = base_query.where(Patient.city.ilike(f"%{city}%"))
        if smoker is not None:
            base_query = base_query.where(Patient.smoker == smoker)
        if min_age is not None:
            base_query = base_query.where(Patient.age >= min_age)
        if max_age is not None:
            base_query = base_query.where(Patient.age <= max_age)
        
        # Count total matching records (for pagination metadata)
        count_query = select(func.count()).select_from(base_query.subquery())
        total = (await self.session.execute(count_query)).scalar_one()
        
        # Fetch paginated results
        patients_query = base_query.offset(skip).limit(limit).order_by(Patient.created_at.desc())
        result = await self.session.execute(patients_query)
        patients = list(result.scalars().all())
        
        return patients, total
    
    async def create(self, **kwargs) -> Patient:
        """
        Create a new patient record.
        
        The session.add() adds the object to the session (like a staging area).
        session.flush() sends the INSERT to the database but doesn't commit yet.
        session.refresh() reloads the object from DB (to get auto-generated values
        like the ID and created_at timestamp).
        
        The actual COMMIT happens in get_db() after the endpoint function returns.
        """
        # Generate a unique external ID
        external_id = f"P-{uuid.uuid4().hex[:8].upper()}"
        patient = Patient(external_id=external_id, **kwargs)
        
        self.session.add(patient)     # stage the INSERT
        await self.session.flush()    # send to DB (but not committed yet)
        await self.session.refresh(patient)  # reload to get DB-generated values
        return patient
    
    async def update(self, patient: Patient, **kwargs) -> Patient:
        """Update specific fields of an existing patient."""
        for key, value in kwargs.items():
            if value is not None:    # only update provided fields
                setattr(patient, key, value)
        
        await self.session.flush()
        await self.session.refresh(patient)
        return patient
    
    async def soft_delete(self, patient: Patient) -> None:
        """Mark patient as deleted without removing the row."""
        patient.is_deleted = True
        await self.session.flush()
```

---

## Step 6: Router — Connecting Everything Together

```python title="app/routers/patients.py"
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from core.database import get_db
from repositories.patient import PatientRepository
from schemas.patient import PatientCreate, PatientUpdate, PatientResponse, PaginatedPatients

router = APIRouter(prefix="/patients", tags=["patients"])

def get_repo(db: AsyncSession = Depends(get_db)) -> PatientRepository:
    """
    Dependency that creates a PatientRepository with the current DB session.
    This connects the request's database session to the repository.
    """
    return PatientRepository(db)

@router.get("", response_model=PaginatedPatients)
async def list_patients(
    city: Optional[str] = Query(None),
    smoker: Optional[bool] = Query(None),
    min_age: Optional[int] = Query(None, ge=0),
    max_age: Optional[int] = Query(None, le=150),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    repo: PatientRepository = Depends(get_repo),
):
    """List patients with optional filtering and pagination."""
    skip = (page - 1) * page_size
    patients, total = await repo.search(
        city=city, smoker=smoker,
        min_age=min_age, max_age=max_age,
        skip=skip, limit=page_size
    )
    
    total_pages = (total + page_size - 1) // page_size
    
    return PaginatedPatients(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        data=[PatientResponse.model_validate(p) for p in patients]
    )

@router.get("/{external_id}", response_model=PatientResponse)
async def get_patient(
    external_id: str,
    repo: PatientRepository = Depends(get_repo),
):
    patient = await repo.get_by_external_id(external_id)
    if not patient:
        raise HTTPException(404, f"Patient '{external_id}' not found")
    return PatientResponse.model_validate(patient)

@router.post("", response_model=PatientResponse, status_code=201)
async def create_patient(
    data: PatientCreate,
    repo: PatientRepository = Depends(get_repo),
):
    patient = await repo.create(**data.model_dump())
    return PatientResponse.model_validate(patient)

@router.patch("/{external_id}", response_model=PatientResponse)
async def update_patient(
    external_id: str,
    updates: PatientUpdate,
    repo: PatientRepository = Depends(get_repo),
):
    patient = await repo.get_by_external_id(external_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    
    # exclude_unset=True: only fields the client actually sent
    changed = updates.model_dump(exclude_unset=True)
    patient = await repo.update(patient, **changed)
    return PatientResponse.model_validate(patient)

@router.delete("/{external_id}", status_code=204)
async def delete_patient(
    external_id: str,
    repo: PatientRepository = Depends(get_repo),
):
    patient = await repo.get_by_external_id(external_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    await repo.soft_delete(patient)
    from fastapi import Response
    return Response(status_code=204)
```

---

## Step 7: Database Migrations with Alembic

Alembic tracks and applies changes to your database schema over time. Think of it as "git for your database schema."

```bash
# Initialize Alembic (only do this once per project)
alembic init alembic

# Generate a migration by comparing your ORM models to the current DB schema
# --autogenerate: Alembic figures out what changed
# -m "...": human-readable description of this migration
alembic revision --autogenerate -m "create_patients_table"

# This creates: alembic/versions/abc123_create_patients_table.py
# Review this file — make sure it looks right!

# Apply all pending migrations to your database
alembic upgrade head

# Roll back the last migration (if something went wrong)
alembic downgrade -1

# See what migrations have been applied
alembic current

# See all migrations
alembic history --verbose
```

Configure `alembic/env.py` to use your settings:

```python title="alembic/env.py"
from core.config import get_settings
from models.base import Base

# Import ALL your models so Alembic can detect them
from models.patient import Patient
# from models.prediction_log import PredictionLog  ← add more as you create them

settings = get_settings()

# Tell Alembic to use our database URL
config.set_main_option("sqlalchemy.url", settings.sync_database_url)

# Tell Alembic what schema to compare against
target_metadata = Base.metadata
```

---

## Running the Full Stack with Docker Compose

```yaml title="docker-compose.yml"
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: mlapi
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - pg-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 5

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_DB: mlapi
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    depends_on:
      postgres:
        condition: service_healthy    # wait until DB is ready
    command: >
      sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"

volumes:
  pg-data:
```

```bash
docker compose up -d
# DB migrations run automatically before the API starts
```

---

## Q&A

**Q: What's the difference between `session.add()`, `session.flush()`, and `session.commit()`?**

- `session.add(obj)` — stages the object in memory (no database yet)
- `session.flush()` — sends the SQL to the database but doesn't finalize it; the transaction is still open; you can still roll back. Use this to get auto-generated IDs before commit.
- `session.commit()` — finalizes all changes permanently; can't roll back after this

In our setup, `session.commit()` happens automatically in `get_db()` after the endpoint returns.

**Q: Why do we use `expire_on_commit=False`?**

By default, after `session.commit()`, SQLAlchemy marks all loaded objects as "expired" — meaning any attribute access triggers a new database query to reload the value. In async code, this is dangerous because the session may be closed by the time you access the attribute. `expire_on_commit=False` keeps the values in memory after commit.

**Q: Should I use `session.get()` or `select()` for lookups?**

`session.get(Patient, pk)` uses the **identity map** — SQLAlchemy's in-memory cache. If you already loaded this patient in the same session, it returns the cached version without hitting the database. `select()` always queries the database. Use `session.get()` for primary key lookups, `select()` for everything else.

**Q: Is Alembic required? Can I use `Base.metadata.create_all()`?**

`create_all()` creates tables that don't exist yet. But it never modifies existing tables. If you add a column to your ORM model, `create_all()` does nothing — the new column doesn't appear in the database. Alembic generates migration scripts that `ALTER TABLE` existing tables. Always use Alembic in production.

**Q: How many database connections should I create (pool_size)?**

Rule of thumb: `pool_size = workers × 2`. With 4 Uvicorn workers, use `pool_size=8`. More connections mean more memory usage on both the app server and PostgreSQL. Check with `SELECT count(*) FROM pg_stat_activity` on PostgreSQL to see actual usage.

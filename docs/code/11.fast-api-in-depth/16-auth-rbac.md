---
id: auth-rbac
title: "16 · Authentication, JWT & Role-Based Access Control"
sidebar_label: "16 · Auth & RBAC"
sidebar_position: 16
tags: [auth, jwt, oauth2, rbac, security, permissions, intermediate]
---

# Authentication, JWT & Role-Based Access Control

> **Advanced Topic** — Securing your ML API so only the right people can do the right things.

---

## Visual Reference

![Role-based access control diagram](https://commons.wikimedia.org/wiki/Special:Redirect/file/Role-based_access_control.svg)

Source: [Wikimedia Commons - Role-based access control](https://commons.wikimedia.org/wiki/File:Role-based_access_control.svg)

## Authentication vs Authorization — Understanding the Difference

These two words are often confused. They mean very different things:

**Authentication** = "Who are you?"
Verifying identity. Like showing your passport at the airport. You prove you are who you claim to be by providing something only you know (password) or have (token).

**Authorization** = "What are you allowed to do?"
Checking permissions. Like having a boarding pass only for economy class — even after proving your identity, you can't sit in business class. Your role determines what you can access.

```
User logs in with username/password  ← Authentication
Server verifies credentials
Server issues JWT token

User requests GET /admin/models      ← Authorization
Server checks JWT: role="api_user"
Server rejects: role must be "admin"
```

In a real ML API:
- All team members authenticate (login)
- Data scientists can trigger training and view all predictions
- Analysts can only view aggregated data
- API clients (mobile apps) can only call `/predict`

---

## Why JWT (JSON Web Tokens)?

The alternative to JWT is **session-based auth**: the server stores user state in memory or a database. Every request does a database lookup to find the session.

JWT takes a different approach: **stateless auth**. The token itself contains the user information. The server doesn't store anything — it just verifies the token's cryptographic signature.

```
Session-based (stateful):
Client sends: session_id=abc123
Server looks up: SELECT * FROM sessions WHERE id='abc123'
→ Requires DB query on every request

JWT (stateless):
Client sends: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Server decodes: {sub: "ravi", role: "data_scientist", exp: 1234567890}
Verifies signature (cryptographic check, no DB needed)
→ No DB query needed
```

**JWT Structure:**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9  ← Header (base64): {"alg": "HS256", "typ": "JWT"}
.
eyJzdWIiOiJyYXZpIiwicm9sZSI6ImRzIiwiZXhwIjoxNzMwMDAwMDAwfQ  ← Payload (base64): {"sub": "ravi", "role": "ds", "exp": ...}
.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c  ← Signature (HMAC-SHA256)
```

The signature is created using your secret key: `HMAC_SHA256(header + "." + payload, secret_key)`. Without knowing the secret key, it's computationally impossible to forge a valid signature.

**JWT is not encrypted** — the payload is just base64 encoded (anyone can decode it). Never put sensitive data (passwords, credit card numbers) in JWT payload. It's safe for user identity, roles, and permissions.

---

## Installation

```bash
pip install python-jose[cryptography]   # JWT encoding/decoding
pip install passlib[bcrypt]             # password hashing
```

---

## Step 1: The User Model

```python title="app/models/user.py"
from sqlalchemy import String, Boolean, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
import enum
from .base import Base, TimestampMixin

class Role(str, Enum):
    """
    User roles in the ML platform.
    'str' mixin: role.value gives the string value directly.
    
    admin:          Can manage users, reload models, view everything
    data_scientist: Can run experiments, view all predictions, manage patients
    analyst:        Read-only access to aggregated data and reports
    api_user:       Can only call /predict — for external integrations
    """
    admin = "admin"
    data_scientist = "data_scientist"
    analyst = "analyst"
    api_user = "api_user"

class User(Base, TimestampMixin):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(200), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(200))
    role: Mapped[Role] = mapped_column(SAEnum(Role), default=Role.api_user)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

---

## Step 2: Password Hashing — Never Store Plain Passwords

```python title="app/core/security.py"
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from core.config import get_settings

settings = get_settings()

# bcrypt is the gold standard for password hashing.
# It's intentionally slow (to slow down brute-force attacks).
# "deprecated='auto'" means older hash algorithms are auto-upgraded on login.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain_password: str) -> str:
    """
    Hash a plain-text password using bcrypt.
    
    bcrypt automatically generates a random salt and incorporates it.
    Two calls with the same password produce different hashes (due to the salt).
    This prevents rainbow table attacks.
    
    Example:
        hash_password("mypassword123")
        → "$2b$12$Uw9Q/mE3k..."  ← this changes every call
    """
    return pwd_context.hash(plain_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check if a plain password matches a stored hash.
    
    bcrypt extracts the salt from the stored hash and re-hashes
    the plain password with that salt, then compares.
    """
    return pwd_context.verify(plain_password, hashed_password)


# ─── JWT Constants ────────────────────────────────────────────────
ALGORITHM = "HS256"               # HMAC with SHA-256
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # short-lived: 15 minutes
REFRESH_TOKEN_EXPIRE_DAYS = 7     # long-lived: 7 days


def create_access_token(data: dict) -> str:
    """
    Create a short-lived JWT access token.
    
    The token contains:
    - sub (subject): username of the authenticated user
    - role: user's role for authorization checks
    - exp: expiration time (15 minutes from now)
    - type: "access" so we can distinguish from refresh tokens
    
    Signed with the JWT_SECRET_KEY from settings.
    """
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload["type"] = "access"
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    """
    Create a long-lived JWT refresh token.
    Refresh tokens are used to get new access tokens without re-entering password.
    They should be stored securely by the client (e.g., httpOnly cookie).
    """
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload["type"] = "refresh"
    # Use a DIFFERENT secret key for refresh tokens
    # If access token secret is compromised, refresh tokens are still safe
    return jwt.encode(payload, settings.jwt_refresh_secret_key, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT access token.
    Raises ValueError if the token is invalid, expired, or tampered with.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise ValueError("Not an access token")
        return payload
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}")
```

---

## Step 3: Auth Dependencies for Endpoints

```python title="app/core/dependencies.py"
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from core.database import get_db
from core.security import decode_access_token
from models.user import User, Role

# OAuth2PasswordBearer tells FastAPI/Swagger: 
# "This endpoint expects a Bearer token in the Authorization header"
# tokenUrl: where clients can get a token (for Swagger UI's "Authorize" button)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Dependency that validates a JWT token and returns the current user.
    
    FastAPI injects this into your endpoint when you write:
    current_user: User = Depends(get_current_user)
    
    If the token is missing, invalid, or expired → 401 Unauthorized.
    If the user account is disabled → 401 Unauthorized.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},   # tells client to use Bearer auth
    )
    
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except ValueError:
        raise credentials_exception
    
    # Look up the user in the database
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise credentials_exception
    
    return user


def require_role(*allowed_roles: Role):
    """
    Factory function that creates a role-checking dependency.
    
    Usage:
        @app.get("/admin")
        async def admin_only(user: User = Depends(require_role(Role.admin))):
            ...
    
    Why a factory? Because we need to pass arguments to the dependency.
    FastAPI dependencies can't receive arguments directly, so we wrap them.
    """
    async def check_role(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}. "
                       f"Your role: {current_user.role.value}"
            )
        return current_user
    return check_role


# ─── Convenient shortcuts ─────────────────────────────────────────
# These are the dependencies you'll actually use in your endpoints

require_admin = require_role(Role.admin)
require_ds_or_admin = require_role(Role.admin, Role.data_scientist)
require_analyst_or_above = require_role(Role.admin, Role.data_scientist, Role.analyst)
require_any_user = get_current_user   # just needs valid token
```

---

## Step 4: Auth Router — Login, Register, Refresh

```python title="app/routers/auth.py"
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr, Field

from core.database import get_db
from core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_access_token
)
from models.user import User, Role

router = APIRouter(prefix="/auth", tags=["authentication"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, description="At least 8 characters")

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900   # 15 minutes in seconds

class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/register", status_code=201)
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Create a new user account.
    By default, new users get the 'api_user' role (least privilege).
    An admin must manually promote users to higher roles.
    """
    # Check if username or email already exists
    existing = await db.execute(
        select(User).where(
            (User.username == data.username) | (User.email == data.email)
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, "Username or email already registered")
    
    user = User(
        username=data.username,
        email=data.email,
        hashed_password=hash_password(data.password),
        role=Role.api_user,   # default: least privilege
    )
    db.add(user)
    await db.flush()
    
    return {"message": "Account created", "username": user.username, "role": user.role}


@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate and receive JWT tokens.
    
    Returns two tokens:
    - access_token: short-lived (15 min), used for API calls
    - refresh_token: long-lived (7 days), used only to get new access tokens
    
    The client should store the refresh_token securely and use the access_token
    for all API requests. When access_token expires, call /auth/refresh.
    """
    result = await db.execute(select(User).where(User.username == data.username))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(data.password, user.hashed_password):
        # Use the same error message for wrong username AND wrong password.
        # Different messages would reveal whether a username exists (user enumeration attack).
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    if not user.is_active:
        raise HTTPException(400, "Account is disabled. Contact an administrator.")
    
    token_data = {"sub": user.username, "role": user.role.value}
    
    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """
    Get a new access token using a refresh token.
    Called when the access token has expired (15 min) without requiring re-login.
    
    Also issues a new refresh token (refresh token rotation).
    Rotation means: if a refresh token is stolen, it can only be used once.
    After the attacker uses it, the legitimate user's use will fail → detects theft.
    """
    try:
        payload = decode_access_token.__wrapped__(data.refresh_token)
        # Note: We need to decode as refresh token, not access token
    except:
        raise HTTPException(401, "Invalid or expired refresh token")
    
    username = payload.get("sub")
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(401, "User not found or inactive")
    
    token_data = {"sub": user.username, "role": user.role.value}
    
    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),  # new refresh token
    )


@router.get("/me")
async def get_my_profile(current_user: User = Depends(get_current_user)):
    """Return the current user's profile. Useful for clients to verify their token."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
    }
```

---

## Step 5: Applying Auth to Your Endpoints

```python title="app/routers/predictions.py"
from fastapi import APIRouter, Depends
from core.dependencies import require_any_user, require_ds_or_admin, require_admin
from models.user import User

router = APIRouter(prefix="/predict", tags=["predictions"])

# Any authenticated user can call /predict
@router.post("")
async def predict(
    data: InsuranceInput,
    current_user: User = Depends(require_any_user),
):
    # current_user contains the logged-in user's info
    # You can use current_user.username for audit logging
    ...

# Only data scientists and admins can see all predictions
@router.get("/history")
async def prediction_history(
    current_user: User = Depends(require_ds_or_admin),
):
    ...

# Only admins can reload the model
@router.post("/reload-model")
async def reload_model(
    current_user: User = Depends(require_admin),
):
    ...
```

---

## The Permission Matrix

```
Endpoint                        api_user  analyst  data_scientist  admin
──────────────────────────────────────────────────────────────────────────
POST  /predict                     ✅        ✅          ✅           ✅
GET   /predict/history             ❌        ✅          ✅           ✅
GET   /patients                    ❌        ✅          ✅           ✅
POST  /patients                    ❌        ❌          ✅           ✅
DELETE /patients/{id}              ❌        ❌          ❌           ✅
POST  /predict/reload-model        ❌        ❌          ❌           ✅
GET   /users                       ❌        ❌          ❌           ✅
PATCH /users/{id}/role             ❌        ❌          ❌           ✅
```

---

## Q&A

**Q: How long should access tokens last?**

A balance between security and user experience. 15 minutes is the industry standard — short enough to limit damage if stolen, long enough for typical interactions. Use refresh tokens so users don't need to re-enter passwords every 15 minutes.

**Q: Is the JWT payload private? Can anyone see what's in it?**

No — JWT is encoded (base64), not encrypted. Anyone who has the token can decode the payload. Never put sensitive data in JWT (passwords, SSNs, credit card numbers). Only put identity information you're comfortable making readable (username, role, user ID).

**Q: What happens when an access token is stolen?**

With standard JWT: the attacker can use it until it expires (15 minutes). This is why access tokens are short-lived. For extra security, implement token blacklisting: store invalidated token IDs in Redis and check on every request.

**Q: Should I store JWT in localStorage or cookies?**

- **localStorage**: vulnerable to XSS (JavaScript can read it)
- **httpOnly cookie**: JavaScript can't read it (safer), but vulnerable to CSRF
- **For ML APIs**: usually API keys or Bearer tokens stored in memory (for mobile/desktop clients), or httpOnly cookies (for web frontends)

**Q: What's the difference between OAuth2 and what we implemented?**

We implemented "OAuth2 with Password Flow" — the user enters credentials directly into your app. True OAuth2 is for "Login with Google/GitHub" — a third-party handles authentication and sends your app a token. For an internal ML platform, the Password Flow is perfectly appropriate.

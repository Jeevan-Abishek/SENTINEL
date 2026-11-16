from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from jose import jwt, JWTError

from app.config import settings
from app.database.base import get_db
from app.database.repositories import OperatorRepository
from app.database.models import Operator


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with get_db() as session:
        yield session


async def get_current_operator(
    token: str = Depends(lambda: "mock-token"),  # Placeholder for OAuth2
) -> Operator:
    """Dependency to get current authenticated operator."""
    # In production, this would validate JWT token
    # For now, return a mock operator
    async with OperatorRepository() as repo:
        operator = await repo.get_by_username("admin")
        if not operator:
            # Create default admin if not exists
            operator = await repo.create(
                username="admin",
                email="admin@sentinel.local",
                hashed_password="hashed_password_placeholder",
                role="admin",
            )
        return operator


def require_role(*allowed_roles: str):
    """Dependency factory for role-based access control."""
    async def role_checker(operator: Operator = Depends(get_current_operator)) -> Operator:
        if operator.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return operator
    return role_checker
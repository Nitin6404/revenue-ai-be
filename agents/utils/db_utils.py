"""
Database utilities for the application.

This module provides functions for managing database connections,
executing queries, and handling database operations in a consistent way.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Callable
from contextlib import contextmanager

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine, Connection, CursorResult
from sqlalchemy.orm import sessionmaker, Session, scoped_session

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')

# Global engine and session factory
_engine: Optional[Engine] = None
_session_factory: Optional[scoped_session] = None

def get_engine() -> Engine:
    """Get the database engine, creating it if it doesn't exist."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=settings.DEBUG,
            future=True
        )
        logger.info(f"Database engine created for {settings.DATABASE_URL}")
    return _engine

def get_session_factory() -> scoped_session:
    """Get the database session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=get_engine(),
                expire_on_commit=True
            )
        )
    return _session_factory

@contextmanager
def get_db() -> Session:
    """Context manager for database sessions."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}", exc_info=True)
        raise
    finally:
        session.close()

def execute_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    connection: Optional[Connection] = None
) -> CursorResult:
    """Execute a raw SQL query."""
    params = params or {}
    try:
        if connection:
            return connection.execute(text(query), params)
        with get_engine().connect() as conn:
            return conn.execute(text(query), params)
    except Exception as e:
        logger.error(f"Error executing query: {query}\nParams: {params}\nError: {str(e)}")
        raise

def fetch_all(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    connection: Optional[Connection] = None
) -> List[Dict[str, Any]]:
    """Fetch all rows from a query as dictionaries."""
    result = execute_query(query, params, connection)
    return [dict(row._mapping) for row in result]

def fetch_one(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    connection: Optional[Connection] = None
) -> Optional[Dict[str, Any]]:
    """Fetch a single row from a query as a dictionary."""
    result = execute_query(query, params, connection)
    row = result.first()
    return dict(row._mapping) if row else None

def table_exists(table_name: str, schema: Optional[str] = None) -> bool:
    """Check if a table exists in the database."""
    return inspect(get_engine()).has_table(table_name, schema=schema)

def get_table_columns(
    table_name: str, 
    schema: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get information about the columns in a table."""
    return inspect(get_engine()).get_columns(table_name, schema=schema)

def bulk_insert(
    table_name: str,
    data: List[Dict[str, Any]],
    connection: Optional[Connection] = None
) -> int:
    """Perform a bulk insert of multiple rows."""
    if not data:
        return 0
    
    columns = list(data[0].keys())
    placeholders = ", ".join([":%s" % col for col in columns])
    column_list = ", ".join([f'"{col}"' for col in columns])
    query = f'INSERT INTO "{table_name}" ({column_list}) VALUES ({placeholders})'
    
    if connection:
        result = connection.execute(text(query), data)
    else:
        with get_engine().begin() as conn:
            result = conn.execute(text(query), data)
    
    return result.rowcount

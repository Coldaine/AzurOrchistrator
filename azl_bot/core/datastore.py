"""Database operations and models."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

Base = declarative_base()


class Run(Base):
    """Database model for task runs."""
    __tablename__ = 'runs'
    
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, nullable=False)
    task = Column(String(50), nullable=False)
    device_serial = Column(String(100), nullable=False)
    
    # Relationship to actions
    actions = relationship("Action", back_populates="run")


class Action(Base):
    """Database model for individual actions."""
    __tablename__ = 'actions'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'), nullable=False)
    ts = Column(DateTime, nullable=False)
    screen = Column(String(50))
    action = Column(String(20), nullable=False)
    selector_json = Column(Text)
    method = Column(String(20))
    point_norm_x = Column(Float)
    point_norm_y = Column(Float)
    confidence = Column(Float)
    success = Column(Boolean)
    
    # Relationship
    run = relationship("Run", back_populates="actions")


class Currency(Base):
    """Database model for currency records."""
    __tablename__ = 'currencies'
    
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, nullable=False)
    oil = Column(Integer)
    coins = Column(Integer)
    gems = Column(Integer)
    cubes = Column(Integer)


class Commission(Base):
    """Database model for commission records."""
    __tablename__ = 'commissions'
    
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, nullable=False)
    slot_id = Column(Integer, nullable=False)
    name = Column(String(200))
    rarity = Column(String(20))
    time_remaining_s = Column(Integer)
    status = Column(String(20))  # "idle" | "in_progress" | "ready"


class DataStore:
    """Database operations manager."""
    
    def __init__(self, db_path: Path) -> None:
        """Initialize datastore.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = None
        self.session_factory = None
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.session_factory()
    
    def insert_run(self, task: str, device_serial: str) -> int:
        """Insert a new task run record.
        
        Args:
            task: Task name
            device_serial: Device serial number
            
        Returns:
            Run ID
        """
        with self.get_session() as session:
            run = Run(
                started_at=datetime.utcnow(),
                task=task,
                device_serial=device_serial
            )
            session.add(run)
            session.commit()
            
            run_id = run.id
            logger.info(f"Created run {run_id} for task '{task}' on device '{device_serial}'")
            return run_id
    
    def append_action(self, run_id: int, screen: Optional[str], action: str, 
                     selector_json: Optional[str] = None, method: Optional[str] = None,
                     point_norm_x: Optional[float] = None, point_norm_y: Optional[float] = None,
                     confidence: Optional[float] = None, success: Optional[bool] = None) -> None:
        """Append an action record to a run.
        
        Args:
            run_id: Run ID
            screen: Screen identifier
            action: Action type
            selector_json: JSON representation of selector
            method: Resolution method used
            point_norm_x: Normalized X coordinate
            point_norm_y: Normalized Y coordinate  
            confidence: Confidence score
            success: Whether action was successful
        """
        with self.get_session() as session:
            action_record = Action(
                run_id=run_id,
                ts=datetime.utcnow(),
                screen=screen,
                action=action,
                selector_json=selector_json,
                method=method,
                point_norm_x=point_norm_x,
                point_norm_y=point_norm_y,
                confidence=confidence,
                success=success
            )
            session.add(action_record)
            session.commit()
            
        logger.debug(f"Logged action: {action} on {screen} (success={success})")
    
    def record_currencies(self, oil: Optional[int] = None, coins: Optional[int] = None,
                         gems: Optional[int] = None, cubes: Optional[int] = None) -> None:
        """Record currency balances.
        
        Args:
            oil: Oil amount
            coins: Coins amount
            gems: Gems amount
            cubes: Cubes amount (optional)
        """
        with self.get_session() as session:
            currency = Currency(
                ts=datetime.utcnow(),
                oil=oil,
                coins=coins,
                gems=gems,
                cubes=cubes
            )
            session.add(currency)
            session.commit()
            
        logger.info(f"Recorded currencies: Oil={oil}, Coins={coins}, Gems={gems}, Cubes={cubes}")
    
    def record_commissions(self, commissions: List[Dict[str, Any]]) -> None:
        """Record commission data.
        
        Args:
            commissions: List of commission dictionaries
        """
        ts = datetime.utcnow()
        
        with self.get_session() as session:
            for commission_data in commissions:
                commission = Commission(
                    ts=ts,
                    slot_id=commission_data.get('slot_id'),
                    name=commission_data.get('name'),
                    rarity=commission_data.get('rarity'),
                    time_remaining_s=commission_data.get('time_remaining_s'),
                    status=commission_data.get('status')
                )
                session.add(commission)
            
            session.commit()
            
        logger.info(f"Recorded {len(commissions)} commission entries")
    
    def get_latest_currencies(self) -> Optional[Currency]:
        """Get the most recent currency record.
        
        Returns:
            Latest currency record or None
        """
        with self.get_session() as session:
            currency = session.query(Currency).order_by(Currency.ts.desc()).first()
            if currency:
                # Detach from session
                session.expunge(currency)
            return currency
    
    def get_latest_commissions(self) -> List[Commission]:
        """Get the most recent commission records.
        
        Returns:
            List of latest commission records
        """
        with self.get_session() as session:
            # Get the latest timestamp
            latest_ts_subquery = session.query(Commission.ts).order_by(Commission.ts.desc()).limit(1).subquery()
            
            # Get all commissions from that timestamp
            commissions = session.query(Commission).filter(
                Commission.ts == latest_ts_subquery.c.ts
            ).order_by(Commission.slot_id).all()
            
            # Detach from session
            for commission in commissions:
                session.expunge(commission)
                
            return commissions
    
    def get_run_actions(self, run_id: int) -> List[Action]:
        """Get all actions for a specific run.
        
        Args:
            run_id: Run ID
            
        Returns:
            List of action records
        """
        with self.get_session() as session:
            actions = session.query(Action).filter(Action.run_id == run_id).order_by(Action.ts).all()
            
            # Detach from session
            for action in actions:
                session.expunge(action)
                
            return actions
    
    def get_recent_runs(self, limit: int = 10) -> List[Run]:
        """Get recent task runs.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of recent run records
        """
        with self.get_session() as session:
            runs = session.query(Run).order_by(Run.started_at.desc()).limit(limit).all()
            
            # Detach from session
            for run in runs:
                session.expunge(run)
                
            return runs
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data beyond retention period.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with self.get_session() as session:
            # Delete old actions
            old_actions = session.query(Action).filter(Action.ts < cutoff_date).count()
            session.query(Action).filter(Action.ts < cutoff_date).delete()
            
            # Delete old runs (that no longer have actions)
            old_runs = session.query(Run).filter(
                Run.started_at < cutoff_date,
                ~Run.actions.any()
            ).count()
            session.query(Run).filter(
                Run.started_at < cutoff_date,
                ~Run.actions.any()
            ).delete()
            
            # Delete old currencies
            old_currencies = session.query(Currency).filter(Currency.ts < cutoff_date).count()
            session.query(Currency).filter(Currency.ts < cutoff_date).delete()
            
            # Delete old commissions
            old_commissions = session.query(Commission).filter(Commission.ts < cutoff_date).count()
            session.query(Commission).filter(Commission.ts < cutoff_date).delete()
            
            session.commit()
            
        logger.info(f"Cleaned up old data: {old_actions} actions, {old_runs} runs, "
                   f"{old_currencies} currencies, {old_commissions} commissions")


def init_migrations(db_path: Path) -> None:
    """Initialize database with migration script.
    
    Args:
        db_path: Path to database file
    """
    # For now, just create the DataStore which will create tables
    # In a full implementation, you'd have proper migration scripts
    datastore = DataStore(db_path)
    logger.info("Database migrations completed")
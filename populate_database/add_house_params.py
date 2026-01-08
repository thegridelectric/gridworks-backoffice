import os
import time
import dotenv
import pendulum
from sqlalchemy import create_engine, text, asc, desc, cast, BigInteger
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import select
from sqlalchemy.exc import OperationalError
from gjk.models import MessageSql
from pydantic import BaseModel  
from typing import Literal, Optional
from math import ceil


class Ha1Params(BaseModel): 
    AlphaTimes10: int
    BetaTimes100: int
    GammaEx6: int
    IntermediatePowerKw: float
    IntermediateRswtF: int
    DdPowerKw: float
    DdRswtF: int
    DdDeltaTF: int
    HpMaxKwTh: Optional[float] = None
    MaxEwtF: Optional[int] = None
    LoadOverestimationPercent: Optional[int] = None
    TypeName: Literal["ha1.params"] = "ha1.params"
    Version: str = "004"


class LayoutLite(BaseModel):
    FromGNodeAlias: Optional[str] = None
    MessageCreatedMs: int
    MessageId: Optional[str] = None
    Strategy: Optional[str] = None
    ZoneList: list[str]
    CriticalZoneList: Optional[list[str]] = None
    TotalStoreTanks: Optional[int] = None
    ShNodes: Optional[list[dict]] = None
    DataChannels: Optional[list[dict]] = None
    SynthChannels: Optional[list[dict]] = None
    TankModuleComponents: Optional[list[dict]] = None
    FlowModuleComponents: Optional[list[dict]] = None
    Ha1Params: Ha1Params
    I2cRelayComponent: Optional[dict] = None
    TypeName: Literal["layout.lite"] = "layout.lite"
    Version: str = "001"


dotenv.load_dotenv()
gbo_db_url = os.getenv("GBO_DB_URL_NO_ASYNC")
gjk_db_url = os.getenv("GJK_DB_URL")
engine_gbo = create_engine(
    gbo_db_url,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,   # Recycle connections after 1 hour
    connect_args={"connect_timeout": 30}
)
engine_gjk = create_engine(
    gjk_db_url,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,   # Recycle connections after 1 hour
    connect_args={"connect_timeout": 30}
)
SessionGbo = sessionmaker(bind=engine_gbo)
SessionGjk = sessionmaker(bind=engine_gjk)

def unix_ms_to_date(time_ms, timezone_str='America/New_York'):
    return str(pendulum.from_timestamp(time_ms/1000, tz=timezone_str).format('YYYY-MM-DD HH:mm'))

def add_parameter_column(parameter_name: str):
    with engine_gbo.connect() as conn:
        # Drop the column if it already exists
        result = conn.execute(text(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='hourly_electricity' AND column_name='{parameter_name}'
        """))
        if result.fetchone():
            print(f"Column '{parameter_name}' exists.")
            return True
            print(f"Dropping it before continuing...")
            conn.execute(text(f"""
                ALTER TABLE hourly_electricity 
                DROP COLUMN {parameter_name}
            """))
            conn.commit()
            print(f"Column '{parameter_name}' dropped.")

        # Add the column
        conn.execute(text(f"""
            ALTER TABLE hourly_electricity 
            ADD COLUMN {parameter_name} FLOAT
        """))
        conn.commit()
        print(f"Column '{parameter_name}' added successfully.")
        return True

def backfill_parameter_values(parameter_name: str):
    print(f"\nBackfilling {parameter_name} values from layout.lite data...")
    session_gbo = SessionGbo()
    session_gjk = SessionGjk()
    try:
        # Finding house aliases that need backfilling for this parameter
        house_result = session_gbo.execute(text(f"""
            SELECT DISTINCT short_alias
            FROM hourly_electricity
            WHERE {parameter_name} IS NULL
            ORDER BY short_alias
        """))
        house_aliases = [row[0] for row in house_result.fetchall()]
        print(f"Adding data for {len(house_aliases)} house(s): {', '.join(house_aliases)}")
        
        # Backfilling values for each house
        for house_alias in house_aliases:
            print(f"\nProcessing {house_alias}...")
            
            # Finding start and end times that need backfilling for this parameter and this house
            result = session_gbo.execute(text(f"""
                SELECT MIN(hour_start_s) as min_hour, MAX(hour_start_s) as max_hour
                FROM hourly_electricity
                WHERE {parameter_name} IS NULL
                AND short_alias = :house_alias
            """), {
                'house_alias': house_alias
            })
            time_range = result.fetchone()
            if not time_range or time_range[0] is None:
                print("No rows need updating.")
                return
            start_ms = time_range[0] * 1000
            end_ms = time_range[1] * 1000
            end_ms = min(int(time.time()*1000), end_ms)
            backfill(session_gbo, session_gjk, house_alias, start_ms, end_ms, parameter_name)
        
    except Exception as e:
        try:
            session_gbo.rollback()
        except Exception:
            pass  # Ignore rollback errors
        print(f"Error during backfill: {e}")
        raise
    finally:
        # Close sessions gracefully, handling already-closed connections
        try:
            session_gbo.close()
        except Exception:
            pass  # Ignore errors when closing
        try:
            session_gjk.close()
        except Exception:
            pass  # Ignore errors when closing

def execute_query_with_retry(session, stmt, max_retries=5, delay=3, backoff=2):
    """Execute a query with retry logic for connection errors."""
    retries = 0
    current_delay = delay
    while retries < max_retries:
        try:
            return session.execute(stmt).scalars().all()
        except OperationalError as e:
            error_str = str(e)
            is_connection_error = (
                "SSL connection has been closed" in error_str or
                "server closed the connection" in error_str.lower() or
                "connection reset" in error_str.lower() or
                "connection" in error_str.lower() and ("closed" in error_str.lower() or "lost" in error_str.lower())
            )
            if is_connection_error:
                retries += 1
                # Always try to rollback to clear the transaction state
                try:
                    session.rollback()
                except Exception:
                    # If rollback fails, try to invalidate the session
                    try:
                        session.invalidate()
                    except Exception:
                        pass  # Ignore if already invalidated
                if retries >= max_retries:
                    raise
                print(f"  Connection error (attempt {retries}/{max_retries}): {error_str[:100]}...")
                print(f"  Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                raise
    return None

def execute_query_first_with_retry(session, stmt, max_retries=5, delay=3, backoff=2):
    """Execute a query and get first result with retry logic for connection errors."""
    retries = 0
    current_delay = delay
    while retries < max_retries:
        try:
            return session.execute(stmt).scalars().first()
        except OperationalError as e:
            error_str = str(e)
            is_connection_error = (
                "SSL connection has been closed" in error_str or
                "server closed the connection" in error_str.lower() or
                "connection reset" in error_str.lower() or
                "connection" in error_str.lower() and ("closed" in error_str.lower() or "lost" in error_str.lower())
            )
            if is_connection_error:
                retries += 1
                # Always try to rollback to clear the transaction state
                try:
                    session.rollback()
                except Exception:
                    # If rollback fails, try to invalidate the session
                    try:
                        session.invalidate()
                    except Exception:
                        pass  # Ignore if already invalidated
                if retries >= max_retries:
                    raise
                print(f"  Connection error (attempt {retries}/{max_retries}): {error_str[:100]}...")
                print(f"  Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                raise
    return None

def backfill(session_gbo: Session, session_gjk: Session, house_alias, start_ms, end_ms, parameter_name):
    print(f"  \nGathering parameters from {unix_ms_to_date(start_ms)} to {unix_ms_to_date(end_ms)}...")
    
    # Find all layouts in the start_ms - end_ms time frame
    stmt = select(MessageSql).filter(
        MessageSql.message_type_name == 'layout.lite',
        MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}.scada",
        MessageSql.message_persisted_ms <= cast(int(end_ms), BigInteger),
        MessageSql.message_persisted_ms >= cast(int(start_ms), BigInteger),
    ).order_by(asc(MessageSql.message_persisted_ms))

    layout_lites: list[MessageSql] = execute_query_with_retry(session_gjk, stmt)
    print(f"  Found {len(layout_lites)} layout lites")

    # Find the last layout before start_ms (if any)
    stmt = select(MessageSql).filter(
        MessageSql.message_type_name == 'layout.lite',
        MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}.scada",
        MessageSql.message_persisted_ms <= cast(int(start_ms), BigInteger),
    ).order_by(desc(MessageSql.message_persisted_ms))
    
    first_layout_lite: MessageSql = execute_query_first_with_retry(session_gjk, stmt)
    if first_layout_lite:
        print(f"  First layout lite: {unix_ms_to_date(first_layout_lite.message_persisted_ms)}")
        all_layout_lites: list[MessageSql] = [first_layout_lite] + layout_lites
    else:
        all_layout_lites: list[MessageSql] = layout_lites

    all_layout_lites = [x for x in all_layout_lites if 'Ha1Params' in x.payload]
    if not all_layout_lites:
        return 0
    
    # Extract the parameter values from the layout lites
    parameter_map = {}
    for l in all_layout_lites:
        ll = LayoutLite(**l.payload)
        params = ll.Ha1Params
        if params is None:
            continue
        time_created = int(ll.MessageCreatedMs/1000)
        if parameter_name == 'alpha':
            parameter_value = params.AlphaTimes10/10
        elif parameter_name == 'beta':
            parameter_value = params.BetaTimes100/100
        elif parameter_name == 'gamma':
            parameter_value = params.GammaEx6/10**6
        elif parameter_name == 'intermediate_power_kw':
            parameter_value = params.IntermediatePowerKw
        elif parameter_name == 'intermediate_rswt':
            parameter_value = params.IntermediateRswtF
        elif parameter_name == 'dd_power_kw':
            parameter_value = params.DdPowerKw
        elif parameter_name == 'dd_rswt':
            parameter_value = params.DdRswtF
        elif parameter_name == 'dd_delta_t':
            parameter_value = params.DdDeltaTF
        else:
            raise ValueError(f"Invalid parameter name: {parameter_name}")
        parameter_map[time_created] = parameter_value

    # Identify the rows to update (only rows where parameter is NULL)
    result = session_gbo.execute(text(f"""
        SELECT g_node_alias, hour_start_s
        FROM hourly_electricity
        WHERE short_alias = :house_alias
        AND hour_start_s >= {int(start_ms/1000)}
        AND hour_start_s < {int(end_ms/1000)}
        AND {parameter_name} IS NULL
        ORDER BY hour_start_s ASC
    """), {
        'house_alias': house_alias
    })
    rows_to_update = result.fetchall()
    print(f"  Found {len(rows_to_update)} rows to update")

    # Update the rows in batches
    batch_size = 100
    num_batches = ceil(len(rows_to_update) / batch_size)

    for batch_idx in range(num_batches):
        print(f"  Processing batch {batch_idx + 1} of {num_batches}")
        updated_count = 0
        batch = rows_to_update[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        
        for row in batch:
            g_node_alias = str(row[0])
            hour_start_s = float(row[1])

            layout_time_created_s = None
            previous_layout_time_created = sorted(parameter_map.keys())[0]
            for layout_time_created in sorted(parameter_map.keys()):
                if hour_start_s < layout_time_created:
                    layout_time_created_s = previous_layout_time_created
                    break
                previous_layout_time_created = layout_time_created

            if layout_time_created_s and layout_time_created_s in parameter_map:
                parameter_value = parameter_map[layout_time_created_s]
                # print(f"  {parameter_name} = {parameter_value} for {unix_ms_to_date(hour_start_s*1000)}")
                session_gbo.execute(text(f"""
                    UPDATE hourly_electricity
                    SET {parameter_name} = {parameter_value}
                    WHERE g_node_alias = :g_node_alias
                    AND hour_start_s = {hour_start_s}
                    AND {parameter_name} IS NULL
                """), {
                    'g_node_alias': g_node_alias,
                })
                updated_count += 1

        session_gbo.commit()
        print(f"  Updated {updated_count} rows in batch {batch_idx + 1} of {num_batches}")
    

if __name__ == '__main__':    
    for parameter_name in ['alpha', 'beta', 'gamma', 'intermediate_power_kw', 'intermediate_rswt', 'dd_power_kw', 'dd_rswt', 'dd_delta_t']:
        print('\n' + "=" * 60 + f" Adding {parameter_name.capitalize()} column " + "=" * 60)
        if add_parameter_column(parameter_name):
            backfill_parameter_values(parameter_name)
    print("\n" + "=" * 60 + " Migration complete! " + "=" * 60)

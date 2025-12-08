import os
import time
import dotenv
import pendulum
from sqlalchemy import create_engine, text, asc, cast, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from gjk.models import MessageSql
from pydantic import BaseModel  


class AtnBid(BaseModel):
    BidderAlias: str
    MarketSlotName: str
    PqPairs: list
    InjectionIsPositive: bool
    PriceUnit: str
    QuantityUnit: str
    SignedMarketFeeTxn: str
    TypeName: str = "atn.bid"
    Version: str = "001"

dotenv.load_dotenv()
gbo_db_url = os.getenv("GBO_DB_URL_NO_ASYNC")
gjk_db_url = os.getenv("GJK_DB_URL")
engine_gbo = create_engine(gbo_db_url)
engine_gjk = create_engine(gjk_db_url)
SessionGbo = sessionmaker(bind=engine_gbo)
SessionGjk = sessionmaker(bind=engine_gjk)

TIMEZONE = 'America/New_York'
SMALL_BATCH_SIZE = 20
LARGE_BATCH_SIZE = 100

def unix_ms_to_date(time_ms, timezone_str):
    return str(pendulum.from_timestamp(time_ms/1000, tz=timezone_str).format('YYYY-MM-DD HH:mm'))

def add_bid_column():
    with engine_gbo.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='hourly_electricity' AND column_name='bid'
        """))
        if result.fetchone():
            print("Column 'bid' exists. Dropping it before continuing...")
            conn.execute(text("""
                ALTER TABLE hourly_electricity 
                DROP COLUMN bid
            """))
            conn.commit()
            print("Column 'bid' dropped.")
            # return False
        
        # Add the column
        conn.execute(text("""
            ALTER TABLE hourly_electricity 
            ADD COLUMN bid VARCHAR(255)
        """))
        conn.commit()
        print("Column 'bid' added successfully.")
        return True

def backfill_bid_values():
    print("\nBackfilling bid values from atn_bids...")
    session_gbo = SessionGbo()
    session_gjk = SessionGjk()
    
    try:
        result = session_gbo.execute(text("""
            SELECT MIN(hour_start_s) as min_hour, MAX(hour_start_s) as max_hour
            FROM hourly_electricity
            WHERE bid IS NULL
        """))
        
        time_range = result.fetchone()
        if not time_range or time_range[0] is None:
            print("No rows need updating.")
            return
        
        min_hour_s = time_range[0]
        max_hour_s = time_range[1]
        print(f"Adding data between {unix_ms_to_date(min_hour_s*1000, TIMEZONE)} and {unix_ms_to_date(max_hour_s*1000, TIMEZONE)}")

        house_result = session_gbo.execute(text("""
            SELECT DISTINCT short_alias
            FROM hourly_electricity
            WHERE bid IS NULL
            ORDER BY short_alias
        """))
        house_aliases = [row[0] for row in house_result.fetchall()]
        print(f"Adding data for {len(house_aliases)} house(s): {', '.join(house_aliases)}")
        
        for house_alias in house_aliases:
            print(f"\nProcessing {house_alias}...")
            backfill_house_bid_values(session_gbo, session_gjk, house_alias, min_hour_s, max_hour_s)
        
    except Exception as e:
        session_gbo.rollback()
        print(f"Error during backfill: {e}")
        raise
    finally:
        session_gbo.close()
        session_gjk.close()

def backfill_house_bid_values(session_gbo, session_gjk, house_alias, min_hour_s, max_hour_s):
    start_ms = min_hour_s * 1000
    end_ms = max_hour_s * 1000
    now_ms = int(time.time() * 1000)
    batch_start_ms = int(pendulum.from_timestamp(start_ms/1000, tz=TIMEZONE).replace(hour=0, minute=0, microsecond=0).timestamp()*1000)
    batch_end_ms = batch_start_ms + (SMALL_BATCH_SIZE+1)*3600*1000
    
    while batch_start_ms < min(end_ms, now_ms):
        batch_end_ms = min(batch_end_ms, now_ms, end_ms)
        try:
            process_batch_for_bid(session_gbo, session_gjk, house_alias, batch_start_ms, batch_end_ms)
        except Exception as e:
            print(f"Error processing batch from {unix_ms_to_date(batch_start_ms, TIMEZONE)} to {unix_ms_to_date(batch_end_ms, TIMEZONE)}: {e}")
        
        batch_start_ms += SMALL_BATCH_SIZE*3600*1000
        batch_end_ms += SMALL_BATCH_SIZE*3600*1000

def extract_pq_pairs(bid: AtnBid):
    if 'PriceTimes1000' in bid.PqPairs[0]:
        return [(pq['PriceTimes1000'] / 1000, pq['QuantityTimes1000'] / 1000) for pq in bid.PqPairs]
    elif 'PriceX1000' in bid.PqPairs[0]:
        return [(pq['PriceX1000'] / 1000, pq['QuantityX1000'] / 1000) for pq in bid.PqPairs]
    else:
        return None

def process_batch_for_bid(session_gbo, session_gjk, house_alias, batch_start_ms, batch_end_ms):
    print(f"  \nGathering bids from {unix_ms_to_date(batch_start_ms, TIMEZONE)} to {unix_ms_to_date(batch_end_ms, TIMEZONE)}...")
    
    stmt = select(MessageSql).filter(
        MessageSql.message_type_name == 'atn.bid',
        MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}",
        MessageSql.message_persisted_ms <= cast(int(batch_end_ms), BigInteger),
        MessageSql.message_persisted_ms >= cast(int(batch_start_ms), BigInteger),
    ).order_by(asc(MessageSql.message_persisted_ms))

    atn_bids: MessageSql = session_gjk.execute(stmt).scalars().all()
    print(f"  Found {len(atn_bids)} atn bids")
    if not atn_bids:
        return 0
    
    bid_map = {}
    for f in atn_bids:
        if 'PqPairs' in f.payload:
            hour_start_s = int((f.message_persisted_ms + 3599_999) // 3_600_000 * 3_600)
            atn_bid = AtnBid(**f.payload)
            pq_pairs = extract_pq_pairs(atn_bid)
            bid_map[hour_start_s] = str(pq_pairs)

    result = session_gbo.execute(text("""
        SELECT g_node_alias, hour_start_s
        FROM hourly_electricity
        WHERE short_alias = :house_alias
        AND hour_start_s >= :batch_start_s
        AND hour_start_s < :batch_end_s
    """), {
        'house_alias': house_alias,
        'batch_start_s': int(batch_start_ms / 1000),
        'batch_end_s': int(batch_end_ms / 1000)
    })
    rows_to_update = result.fetchall()
    print(f"  Found {len(rows_to_update)} rows to update:")
    if not rows_to_update:
        print(f"  Looking for short_alias: {house_alias}, batch_start_s: {int(batch_start_ms / 1000)}, batch_end_s: {int(batch_end_ms / 1000)}")

    updated_count = 0
    for row in rows_to_update:
        g_node_alias = row[0]
        hour_start_s = row[1]
        
        if hour_start_s in bid_map:
            bid_value = bid_map[hour_start_s]
            print(f"  {bid_value}")
            session_gbo.execute(text("""
                UPDATE hourly_electricity
                SET bid = :bid
                WHERE g_node_alias = :g_node_alias
                AND hour_start_s = :hour_start_s
            """), {
                'bid': bid_value,
                'g_node_alias': g_node_alias,
                'hour_start_s': hour_start_s
            })
            updated_count += 1
    
    session_gbo.commit()
    print(f"  Updated {updated_count} rows")
    

if __name__ == '__main__':    
    column_added = add_bid_column()
    if column_added:
        backfill_bid_values()
    print("\n" + "=" * 60 + " Migration complete! " + "=" * 60)

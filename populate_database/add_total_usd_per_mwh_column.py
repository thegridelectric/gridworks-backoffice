"""
Clear and backfill lmp_usd_per_mwh in hourly_electricity from flo.params.house0.
Same logic as add_hourly_data.py: message_type_name == 'flo.params.house0',
lmp_usd_per_mwh = round(LmpForecast[0] + DistPriceForecast[0], 3), matched by
payload['StartUnixS'] == hour_start_s.
"""
import os
import time
import dotenv
import pendulum
from sqlalchemy import create_engine, text, asc, cast, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from sqlalchemy.exc import OperationalError
from gjk.models import MessageSql

dotenv.load_dotenv()
gbo_db_url = os.getenv("GBO_DB_URL_NO_ASYNC")
gjk_db_url = os.getenv("GJK_DB_URL")
engine_gbo = create_engine(gbo_db_url)
engine_gjk = create_engine(gjk_db_url)
SessionGbo = sessionmaker(bind=engine_gbo)
SessionGjk = sessionmaker(bind=engine_gjk)

TIMEZONE = "America/New_York"
# Batch size for querying GJK (hours)
BATCH_HOURS = 500


def unix_ms_to_date(time_ms, timezone_str):
    return str(
        pendulum.from_timestamp(time_ms / 1000, tz=timezone_str).format(
            "YYYY-MM-DD HH:mm"
        )
    )


def create_lmp_usd_per_mwh_column():
    print("Creating lmp_usd_per_mwh column in hourly_electricity...")
    with engine_gbo.connect() as conn:
        # Avoid hanging forever if another session is holding a table lock.
        conn.execute(text("SET lock_timeout = '5s'"))

        column_exists = conn.execute(
            text("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'hourly_electricity'
                  AND column_name = 'lmp_usd_per_mwh'
            )
            """)
        ).scalar()

        if column_exists:
            print("Dropping existing lmp_usd_per_mwh column...")
            try:
                conn.execute(
                    text("""
                    ALTER TABLE hourly_electricity
                    DROP COLUMN lmp_usd_per_mwh
                    """)
                )
            except OperationalError as e:
                # If we can't grab the DDL lock quickly, keep the existing column and proceed.
                print(
                    "Could not drop lmp_usd_per_mwh within 5s due to table lock; "
                    "using existing column instead."
                )
                print(f"Lock detail: {e}")
                conn.rollback()
                return

        conn.execute(
            text("""
            ALTER TABLE hourly_electricity ADD COLUMN lmp_usd_per_mwh FLOAT
            """)
        )
        conn.commit()
        print("Created lmp_usd_per_mwh column in hourly_electricity.")

def clear_lmp_usd_per_mwh():
    print("Clearing lmp_usd_per_mwh for all rows in hourly_electricity...")
    """Set lmp_usd_per_mwh to NULL for all rows in hourly_electricity."""
    with engine_gbo.connect() as conn:
        conn.execute(
            text("""
            UPDATE hourly_electricity
            SET lmp_usd_per_mwh = NULL
            """)
        )
        conn.commit()
        print("Cleared lmp_usd_per_mwh for all rows in hourly_electricity.")


def backfill_lmp_usd_per_mwh():
    """Backfill lmp_usd_per_mwh from flo.params.house0 (LmpForecast + DistPriceForecast)."""
    print("\nBackfilling lmp_usd_per_mwh from flo.params.house0...")
    session_gbo = SessionGbo()
    session_gjk = SessionGjk()

    try:
        result = session_gbo.execute(
            text("""
            SELECT MIN(hour_start_s) AS min_hour, MAX(hour_start_s) AS max_hour
            FROM hourly_electricity
            """)
        )
        time_range = result.fetchone()
        if not time_range or time_range[0] is None:
            print("No rows in hourly_electricity.")
            return

        min_hour_s = time_range[0]
        max_hour_s = time_range[1]
        print(
            f"Filling data between {unix_ms_to_date(min_hour_s * 1000, TIMEZONE)} "
            f"and {unix_ms_to_date(max_hour_s * 1000, TIMEZONE)}"
        )

        # flo.params are matched by StartUnixS (hour start in seconds); query by message_created_ms
        # with slack so we get params for all hours in range
        start_ms = (min_hour_s - 3600) * 1000
        end_ms = (max_hour_s + 7200) * 1000

        stmt = (
            select(MessageSql)
            .filter(
                MessageSql.message_type_name == "flo.params.house0",
                MessageSql.message_created_ms >= cast(int(start_ms), BigInteger),
                MessageSql.message_created_ms <= cast(int(end_ms), BigInteger),
            )
            .order_by(asc(MessageSql.message_created_ms))
        )
        messages = session_gjk.execute(stmt).scalars().all()
        print(f"Found {len(messages)} flo.params.house0 messages")

        # Map hour_start_s -> lmp_usd_per_mwh (same as add_hourly_data: StartUnixS, Lmp+Dist rounded)
        price_by_hour = {}
        for m in messages:
            try:
                hour_start_s = int(m.payload["StartUnixS"])
                lmp = float(m.payload["LmpForecast"][0])
                dist = float(m.payload["DistPriceForecast"][0])
                lmp_usd_per_mwh = round(lmp + dist, 3)
            except (KeyError, IndexError, TypeError):
                continue
            if min_hour_s <= hour_start_s <= max_hour_s:
                price_by_hour[hour_start_s] = lmp_usd_per_mwh

        print(f"Built price map for {len(price_by_hour)} hours")

        # Batch updates (executemany) to avoid thousands of round trips.
        update_stmt = text("""
            UPDATE hourly_electricity
            SET lmp_usd_per_mwh = :lmp_usd_per_mwh
            WHERE hour_start_s = :hour_start_s
        """)

        BATCH_SIZE = 100
        MAX_RETRIES = 4
        updated_count = 0
        items = list(price_by_hour.items())
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch = items[batch_start:batch_start + BATCH_SIZE]
            params = [
                {"hour_start_s": hour_start_s, "lmp_usd_per_mwh": lmp_usd_per_mwh}
                for hour_start_s, lmp_usd_per_mwh in batch
            ]
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    session_gbo.execute(update_stmt, params)
                    session_gbo.commit()
                    updated_count += len(params)
                    print(f"Updated {updated_count}/{len(items)} hours...")
                    break
                except OperationalError as e:
                    print(
                        f"Batch starting at index {batch_start} failed "
                        f"(attempt {attempt}/{MAX_RETRIES}): {e}"
                    )
                    try:
                        session_gbo.rollback()
                    except Exception:
                        pass
                    try:
                        session_gbo.close()
                    except Exception:
                        pass
                    if attempt == MAX_RETRIES:
                        raise
                    # Recreate connection/session and retry this same batch.
                    session_gbo = SessionGbo()
                    backoff_s = min(2 ** attempt, 15)
                    print(f"Retrying this batch in {backoff_s}s...")
                    time.sleep(backoff_s)

        print(f"Updated {updated_count} rows with lmp_usd_per_mwh")

    except Exception as e:
        session_gbo.rollback()
        print(f"Error during backfill: {e}")
        raise
    finally:
        # Close sessions; ignore dead connection errors so we don't raise after successful work
        for name, session in (("gbo", session_gbo), ("gjk", session_gjk)):
            try:
                session.close()
            except (OperationalError, Exception):
                pass


if __name__ == "__main__":
    create_lmp_usd_per_mwh_column()
    clear_lmp_usd_per_mwh()
    backfill_lmp_usd_per_mwh()
    print("\n" + "=" * 60 + " lmp_usd_per_mwh backfill complete " + "=" * 60)

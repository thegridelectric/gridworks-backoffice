"""
Clear and backfill total_usd_per_mwh in hourly_electricity from flo.params.house0.
Same logic as add_hourly_data.py: message_type_name == 'flo.params.house0',
total_usd_per_mwh = round(LmpForecast[0] + DistPriceForecast[0], 3), matched by
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


def clear_total_usd_per_mwh():
    """Set total_usd_per_mwh to NULL for all rows in hourly_electricity."""
    with engine_gbo.connect() as conn:
        conn.execute(
            text("""
            UPDATE hourly_electricity
            SET total_usd_per_mwh = NULL
            """)
        )
        conn.commit()
        print("Cleared total_usd_per_mwh for all rows in hourly_electricity.")


def backfill_total_usd_per_mwh():
    """Backfill total_usd_per_mwh from flo.params.house0 (LmpForecast + DistPriceForecast)."""
    print("\nBackfilling total_usd_per_mwh from flo.params.house0...")
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

        # Map hour_start_s -> total_usd_per_mwh (same as add_hourly_data: StartUnixS, Lmp+Dist rounded)
        price_by_hour = {}
        for m in messages:
            try:
                hour_start_s = int(m.payload["StartUnixS"])
                lmp = float(m.payload["LmpForecast"][0])
                dist = float(m.payload["DistPriceForecast"][0])
                total_usd_per_mwh = round(lmp + dist, 3)
            except (KeyError, IndexError, TypeError):
                continue
            if min_hour_s <= hour_start_s <= max_hour_s:
                price_by_hour[hour_start_s] = total_usd_per_mwh

        print(f"Built price map for {len(price_by_hour)} hours")

        # Update in batches and commit often to avoid long-lived transactions (SSL timeouts)
        COMMIT_EVERY = 200
        updated_count = 0
        for i, (hour_start_s, total_usd_per_mwh) in enumerate(price_by_hour.items()):
            r = session_gbo.execute(
                text("""
                UPDATE hourly_electricity
                SET total_usd_per_mwh = :total_usd_per_mwh
                WHERE hour_start_s = :hour_start_s
                """),
                {
                    "total_usd_per_mwh": total_usd_per_mwh,
                    "hour_start_s": hour_start_s,
                },
            )
            updated_count += r.rowcount
            if (i + 1) % COMMIT_EVERY == 0:
                session_gbo.commit()

        session_gbo.commit()
        print(f"Updated {updated_count} rows with total_usd_per_mwh")

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
    clear_total_usd_per_mwh()
    backfill_total_usd_per_mwh()
    print("\n" + "=" * 60 + " total_usd_per_mwh backfill complete " + "=" * 60)

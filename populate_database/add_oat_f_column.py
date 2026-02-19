"""
Clear and backfill oat_f and ws_mph in hourly_electricity from weather.forecast.
Data: message_type_name == 'weather.forecast',
from_alias == 'hw1.isone.me.versant.keene.{house_alias}.scada',
oat_f <- payload['OatF'][0], ws_mph <- payload['WindSpeedMph'][0].

Uses fresh DB sessions per batch to avoid SSL timeout issues.
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
engine_gbo = create_engine(gbo_db_url, pool_pre_ping=True)
engine_gjk = create_engine(gjk_db_url, pool_pre_ping=True)
SessionGbo = sessionmaker(bind=engine_gbo)
SessionGjk = sessionmaker(bind=engine_gjk)

TIMEZONE = "America/New_York"
BATCH_HOURS = 24  # query this many hours of weather.forecast at a time


def unix_ms_to_date(time_ms, timezone_str):
    return str(
        pendulum.from_timestamp(time_ms / 1000, tz=timezone_str).format(
            "YYYY-MM-DD HH:mm"
        )
    )


def safe_close(session):
    """Close a session, ignoring errors from dropped connections."""
    try:
        session.close()
    except (OperationalError, Exception):
        pass


def clear_oat_f_and_ws_mph():
    """Clear existing oat_f and ws_mph for all rows in hourly_electricity."""
    with engine_gbo.connect() as conn:
        # Drop oat_f_weather if it was added by an earlier run (we now use oat_f only)
        res = conn.execute(
            text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'hourly_electricity' AND column_name = 'oat_f_weather'
            """)
        )
        if res.fetchone():
            conn.execute(text("ALTER TABLE hourly_electricity DROP COLUMN oat_f_weather"))
            conn.commit()
            print("Dropped column oat_f_weather (no longer used).")
        conn.execute(
            text("""
            UPDATE hourly_electricity
            SET oat_f = NULL, ws_mph = NULL
            """)
        )
        conn.commit()
        print("Cleared oat_f and ws_mph for all rows in hourly_electricity.")


def get_time_range_and_houses():
    """Return (min_hour_s, max_hour_s, house_aliases) from hourly_electricity."""
    session_gbo = SessionGbo()
    try:
        result = session_gbo.execute(
            text("""
            SELECT MIN(hour_start_s) AS min_hour, MAX(hour_start_s) AS max_hour
            FROM hourly_electricity
            """)
        )
        time_range = result.fetchone()
        if not time_range or time_range[0] is None:
            return None, None, []

        house_result = session_gbo.execute(
            text("""
            SELECT DISTINCT short_alias
            FROM hourly_electricity
            ORDER BY short_alias
            """)
        )
        house_aliases = [row[0] for row in house_result.fetchall()]
        return time_range[0], time_range[1], house_aliases
    finally:
        safe_close(session_gbo)


def backfill_oat_f_and_ws_mph():
    """Backfill oat_f and ws_mph from weather.forecast (OatF, WindSpeedMph)."""
    print("\nBackfilling oat_f and ws_mph from weather.forecast...")

    min_hour_s, max_hour_s, house_aliases = get_time_range_and_houses()
    if min_hour_s is None:
        print("No rows in hourly_electricity.")
        return

    print(
        f"Filling data between {unix_ms_to_date(min_hour_s * 1000, TIMEZONE)} "
        f"and {unix_ms_to_date(max_hour_s * 1000, TIMEZONE)}"
    )
    print(f"Processing {len(house_aliases)} house(s): {', '.join(house_aliases)}")

    for house_alias in house_aliases:
        print(f"\nProcessing {house_alias}...")
        backfill_house_oat_f_and_ws_mph(house_alias, min_hour_s, max_hour_s)


def backfill_house_oat_f_and_ws_mph(house_alias, min_hour_s, max_hour_s):
    """Backfill oat_f and ws_mph for one house, one small batch at a time."""
    start_s = min_hour_s
    end_s = max_hour_s
    now_s = int(time.time())

    batch_start_s = start_s
    while batch_start_s <= min(end_s, now_s):
        batch_end_s = min(batch_start_s + BATCH_HOURS * 3600, end_s + 1, now_s)
        try:
            process_batch(house_alias, batch_start_s, batch_end_s)
        except Exception as e:
            print(
                f"  Error batch {unix_ms_to_date(batch_start_s * 1000, TIMEZONE)} "
                f"to {unix_ms_to_date(batch_end_s * 1000, TIMEZONE)}: {e}"
            )
        batch_start_s += BATCH_HOURS * 3600


def process_batch(house_alias, batch_start_s, batch_end_s):
    """Fetch weather.forecast for one batch and update hourly_electricity.

    Opens and closes fresh sessions per batch to keep transactions short.
    """
    batch_start_ms = batch_start_s * 1000
    batch_end_ms = batch_end_s * 1000

    # --- query journal DB for weather.forecast messages ---
    session_gjk = SessionGjk()
    try:
        stmt = (
            select(MessageSql)
            .filter(
                MessageSql.message_type_name == "weather.forecast",
                MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}.scada",
                MessageSql.message_created_ms >= cast(int(batch_start_ms), BigInteger),
                MessageSql.message_created_ms <= cast(int(batch_end_ms), BigInteger),
            )
            .order_by(asc(MessageSql.message_created_ms))
        )
        messages = session_gjk.execute(stmt).scalars().all()

        # Build hour_start_s -> (oat_f, ws_mph)
        weather_by_hour = {}
        for m in messages:
            try:
                oat_val = float(m.payload["OatF"][0])
                ws_val = float(m.payload["WindSpeedMph"][0])
            except (KeyError, IndexError, TypeError):
                continue
            hour_start_s = int(m.message_created_ms // 3_600_000) * 3600
            weather_by_hour[hour_start_s] = (oat_val, ws_val)
    finally:
        safe_close(session_gjk)

    if not weather_by_hour:
        return

    # --- update GBO DB ---
    session_gbo = SessionGbo()
    try:
        result = session_gbo.execute(
            text("""
            SELECT g_node_alias, hour_start_s
            FROM hourly_electricity
            WHERE short_alias = :house_alias
            AND hour_start_s >= :batch_start_s
            AND hour_start_s < :batch_end_s
            """),
            {
                "house_alias": house_alias,
                "batch_start_s": batch_start_s,
                "batch_end_s": batch_end_s,
            },
        )
        rows_to_update = result.fetchall()

        updated_count = 0
        for row in rows_to_update:
            g_node_alias, hour_start_s = row[0], row[1]
            if hour_start_s in weather_by_hour:
                oat_f, ws_mph = weather_by_hour[hour_start_s]
                session_gbo.execute(
                    text("""
                    UPDATE hourly_electricity
                    SET oat_f = :oat_f, ws_mph = :ws_mph
                    WHERE g_node_alias = :g_node_alias
                    AND hour_start_s = :hour_start_s
                    """),
                    {
                        "oat_f": oat_f,
                        "ws_mph": ws_mph,
                        "g_node_alias": g_node_alias,
                        "hour_start_s": hour_start_s,
                    },
                )
                updated_count += 1

        session_gbo.commit()
        if updated_count:
            print(
                f"  {unix_ms_to_date(batch_start_ms, TIMEZONE)} - "
                f"{unix_ms_to_date(batch_end_ms, TIMEZONE)}: "
                f"updated {updated_count} rows"
            )
    except Exception:
        session_gbo.rollback()
        raise
    finally:
        safe_close(session_gbo)


if __name__ == "__main__":
    clear_oat_f_and_ws_mph()
    backfill_oat_f_and_ws_mph()
    print("\n" + "=" * 60 + " oat_f / ws_mph backfill complete " + "=" * 60)

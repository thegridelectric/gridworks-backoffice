"""
Add zone1_at_setpoint … zone4_at_setpoint to hourly_electricity (BOOLEAN, nullable).

For each hour, compares mean(zone{x}-*-set) vs mean(zone{x}-*-temp) from report /
batched.readings (same parsing as add_hourly_data.py). Values are degF×1000;
if |avg_set - avg_temp| > 500 (0.5 °F) the column is False, else True. Missing
pair or no samples in the hour → NULL.

Configure HOUSE_ALIAS, START_DATE, END_DATE, and TIMEZONE at the top of this file.
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import pendulum
from sqlalchemy import asc, create_engine, or_, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from gjk.models import MessageSql

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
HOUSE_ALIAS = "beech"
START_DATE = (2025, 11, 1)  # (year, month, day) in TIMEZONE, start of day inclusive
END_DATE = (2026, 4, 1)  # end of this calendar day inclusive
TIMEZONE = "America/New_York"

BATCH_HOURS = 100
AT_SETPOINT_MAX_DIFF_DEGFX1000 = 500
SLACK_MS = 7 * 60 * 1000
UPDATE_BATCH_SIZE = 100
MAX_RETRIES = 4

ZONE_INDICES = (1, 2, 3, 4)

gbo_db_url = os.getenv("GBO_DB_URL_NO_ASYNC") or os.getenv("GBO_DB_URL")
gjk_db_url = os.getenv("GJK_DB_URL")
if gbo_db_url and "asyncpg" in gbo_db_url:
    gbo_db_url = gbo_db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
if gjk_db_url and "asyncpg" in gjk_db_url:
    gjk_db_url = gjk_db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")

engine_gbo = create_engine(gbo_db_url, pool_pre_ping=True)
engine_gjk = create_engine(gjk_db_url, pool_pre_ping=True)
SessionGbo = sessionmaker(bind=engine_gbo)
SessionGjk = sessionmaker(bind=engine_gjk)


def unix_ms_to_date(time_ms: float, timezone_str: str) -> str:
    return str(
        pendulum.from_timestamp(time_ms / 1000, tz=timezone_str).format(
            "YYYY-MM-DD HH:mm"
        )
    )


def safe_close(session) -> None:
    try:
        session.close()
    except (OperationalError, Exception):
        pass


def ensure_zone_at_setpoint_columns() -> None:
    print("Ensuring zoneN_at_setpoint columns on hourly_electricity...")
    with engine_gbo.connect() as conn:
        conn.execute(text("SET lock_timeout = '5s'"))
        for z in ZONE_INDICES:
            col = f"zone{z}_at_setpoint"
            exists = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'hourly_electricity'
                          AND column_name = :col
                    )
                    """
                ),
                {"col": col},
            ).scalar()
            if not exists:
                conn.execute(
                    text(
                        f"ALTER TABLE hourly_electricity "
                        f"ADD COLUMN {col} BOOLEAN"
                    )
                )
                print(f"  Added column {col}")
        conn.commit()
    print("Column check complete.")


def resolve_g_node_aliases() -> List[str]:
    session = SessionGbo()
    try:
        rows = session.execute(
            text(
                """
                SELECT DISTINCT g_node_alias
                FROM hourly_electricity
                WHERE short_alias = :house
                ORDER BY g_node_alias
                """
            ),
            {"house": HOUSE_ALIAS},
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        safe_close(session)


def date_range_to_ms() -> Tuple[int, int]:
    start_ms = int(
        pendulum.datetime(
            START_DATE[0], START_DATE[1], START_DATE[2], tz=TIMEZONE
        )
        .start_of("day")
        .timestamp()
        * 1000
    )
    end_ms = int(
        pendulum.datetime(END_DATE[0], END_DATE[1], END_DATE[2], tz=TIMEZONE)
        .add(days=1)
        .start_of("day")
        .timestamp()
        * 1000
    )
    return start_ms, end_ms


def find_zone_set_temp_pair(
    channel_names: List[str], z: int
) -> Tuple[Optional[str], Optional[str]]:
    prefix = f"zone{z}-"
    set_names = [
        n
        for n in channel_names
        if n.startswith(prefix) and n.endswith("-set")
    ]
    temp_names = [
        n
        for n in channel_names
        if n.startswith(prefix) and n.endswith("-temp")
    ]
    if len(set_names) != 1 or len(temp_names) != 1:
        return None, None
    set_ch, temp_ch = set_names[0], temp_names[0]
    mid_s = set_ch[len(prefix) : -len("-set")]
    mid_t = temp_ch[len(prefix) : -len("-temp")]
    if mid_s != mid_t:
        return None, None
    return set_ch, temp_ch


def mean_value_in_hour(
    times: List[int], values: List[Any], hour_start_ms: int, hour_end_ms: int
) -> Optional[float]:
    nums: List[float] = []
    for t, v in zip(times, values):
        try:
            if hour_start_ms <= int(t) < hour_end_ms:
                nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return None
    return sum(nums) / len(nums)


def zone_at_setpoint_for_hour(
    channels: Dict[str, Dict[str, List]],
    hour_start_ms: int,
    hour_end_ms: int,
) -> Dict[int, Optional[bool]]:
    out: Dict[int, Optional[bool]] = {z: None for z in ZONE_INDICES}
    names = list(channels.keys())
    for z in ZONE_INDICES:
        set_ch, temp_ch = find_zone_set_temp_pair(names, z)
        if not set_ch or not temp_ch:
            continue
        avg_set = mean_value_in_hour(
            channels[set_ch]["times"],
            channels[set_ch]["values"],
            hour_start_ms,
            hour_end_ms,
        )
        avg_temp = mean_value_in_hour(
            channels[temp_ch]["times"],
            channels[temp_ch]["values"],
            hour_start_ms,
            hour_end_ms,
        )
        if avg_set is None or avg_temp is None:
            continue
        out[z] = abs(avg_set - avg_temp) <= AT_SETPOINT_MAX_DIFF_DEGFX1000
    return out


def build_channels_for_messages(
    reports: List[MessageSql],
    hour_start_ms: int,
    hour_end_ms: int,
) -> Dict[str, Dict[str, List]]:
    selected = [
        m
        for m in reports
        if m.message_created_ms >= hour_start_ms - SLACK_MS
        and m.message_created_ms <= hour_end_ms + SLACK_MS
    ]
    channels: Dict[str, Dict[str, List]] = {}
    for message in selected:
        for channel in message.payload["ChannelReadingList"]:
            if message.message_type_name == "report":
                channel_name = channel["ChannelName"]
            elif message.message_type_name == "batched.readings":
                channel_name = None
                for dc in message.payload["DataChannelList"]:
                    if dc["Id"] == channel["ChannelId"]:
                        channel_name = dc["Name"]
                        break
                if channel_name is None:
                    continue
            else:
                continue
            if channel_name not in channels:
                channels[channel_name] = {"times": [], "values": []}
            channels[channel_name]["times"].extend(
                channel["ScadaReadTimeUnixMsList"]
            )
            channels[channel_name]["values"].extend(channel["ValueList"])
    for channel_name in channels:
        pairs = sorted(
            zip(channels[channel_name]["times"], channels[channel_name]["values"])
        )
        if pairs:
            t, v = zip(*pairs)
            channels[channel_name]["times"] = list(t)
            channels[channel_name]["values"] = list(v)
    return channels


def fetch_reports_batch(
    session_gjk, batch_start_ms: int, batch_end_ms: int
) -> List[MessageSql]:
    scada = f"hw1.isone.me.versant.keene.{HOUSE_ALIAS}.scada"
    return (
        session_gjk.query(MessageSql)
        .filter(
            MessageSql.from_alias == scada,
            or_(
                MessageSql.message_type_name == "batched.readings",
                MessageSql.message_type_name == "report",
            ),
            MessageSql.message_created_ms >= batch_start_ms - SLACK_MS,
            MessageSql.message_created_ms <= batch_end_ms + SLACK_MS,
        )
        .order_by(asc(MessageSql.message_created_ms))
        .all()
    )


def backfill_zone_at_setpoint() -> None:
    g_node_aliases = resolve_g_node_aliases()
    if not g_node_aliases:
        print(
            f"No hourly_electricity rows for short_alias={HOUSE_ALIAS!r}; nothing to do."
        )
        return

    start_ms, end_ms = date_range_to_ms()
    now_ms = int(time.time() * 1000)
    cap_ms = min(end_ms, now_ms)

    print(
        f"Backfilling zone at-setpoint for {HOUSE_ALIAS} "
        f"g_node_alias in {g_node_aliases} "
        f"from {unix_ms_to_date(start_ms, TIMEZONE)} "
        f"to {unix_ms_to_date(cap_ms, TIMEZONE)}"
    )

    batch_start_ms = int(
        pendulum.from_timestamp(start_ms / 1000, tz=TIMEZONE)
        .start_of("day")
        .timestamp()
        * 1000
    )

    session_gjk = SessionGjk()
    session_gbo = SessionGbo()

    set_parts = ", ".join(f"zone{z}_at_setpoint = :z{z}" for z in ZONE_INDICES)
    update_sql = text(
        f"""
        UPDATE hourly_electricity
        SET {set_parts}
        WHERE g_node_alias = :g_node_alias
          AND short_alias = :short_alias
          AND hour_start_s = :hour_start_s
        """
    )

    pending: List[dict] = []

    def flush_pending() -> None:
        nonlocal session_gbo, pending
        if not pending:
            return
        n_flush = len(pending)
        print(f"  GBO: flushing {n_flush} UPDATE(s)...", flush=True)
        t_flush = time.monotonic()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                for row in pending:
                    session_gbo.execute(update_sql, row)
                session_gbo.commit()
                print(
                    f"  GBO: committed {n_flush} row(s) in "
                    f"{time.monotonic() - t_flush:.2f}s",
                    flush=True,
                )
                pending = []
                return
            except OperationalError as e:
                print(
                    f"  GBO: batch update failed (attempt {attempt}/{MAX_RETRIES}): {e}",
                    flush=True,
                )
                try:
                    session_gbo.rollback()
                except Exception:
                    pass
                safe_close(session_gbo)
                if attempt == MAX_RETRIES:
                    raise
                session_gbo = SessionGbo()
                time.sleep(min(2**attempt, 15))

    try:
        while batch_start_ms < cap_ms:
            t_batch = time.monotonic()
            batch_end_ms = min(
                batch_start_ms + BATCH_HOURS * 3600 * 1000, cap_ms
            )
            print(
                f"\nBatch {unix_ms_to_date(batch_start_ms, TIMEZONE)} "
                f"→ {unix_ms_to_date(batch_end_ms, TIMEZONE)}",
                flush=True,
            )
            print(
                "  GJK: querying reports (this can take a while)...",
                flush=True,
            )
            t_gjk = time.monotonic()
            reports = fetch_reports_batch(session_gjk, batch_start_ms, batch_end_ms)
            print(
                f"  GJK: loaded {len(reports)} raw messages in "
                f"{time.monotonic() - t_gjk:.2f}s",
                flush=True,
            )
            reports = [
                m
                for m in reports
                if m.message_type_name in ("report", "batched.readings")
            ]
            print(
                f"  GJK: {len(reports)} report/batched.readings after filter",
                flush=True,
            )
            if not reports:
                print("  No reports in batch window.", flush=True)
                print(
                    f"  Batch finished in {time.monotonic() - t_batch:.2f}s (no data)",
                    flush=True,
                )
                batch_start_ms = batch_end_ms
                continue

            hour_start_ms = int(batch_start_ms) - 3600 * 1000
            hour_end_ms = int(batch_start_ms)
            hour_idx = 0
            while hour_end_ms < batch_end_ms:
                hour_start_ms += 3600 * 1000
                hour_end_ms += 3600 * 1000
                if hour_start_ms < start_ms or hour_start_ms >= end_ms:
                    continue

                hour_idx += 1
                t_h = time.monotonic()
                print(
                    f"    hour #{hour_idx} "
                    f"{unix_ms_to_date(hour_start_ms, TIMEZONE)} "
                    f"hour_start_s={hour_start_ms // 1000} …",
                    flush=True,
                )
                chans = build_channels_for_messages(
                    reports, hour_start_ms, hour_end_ms
                )
                t_after_build = time.monotonic()
                flags = zone_at_setpoint_for_hour(chans, hour_start_ms, hour_end_ms)
                t_after_flags = time.monotonic()
                hour_start_s = hour_start_ms // 1000
                print(
                    f"    hour #{hour_idx} "
                    f"channels={len(chans)} "
                    f"build={t_after_build - t_h:.3f}s "
                    f"flags={t_after_flags - t_after_build:.3f}s "
                    f"flags_dict={flags}",
                    flush=True,
                )
                for g_node_alias in g_node_aliases:
                    row = {
                        "g_node_alias": g_node_alias,
                        "short_alias": HOUSE_ALIAS,
                        "hour_start_s": hour_start_s,
                    }
                    for z in ZONE_INDICES:
                        v = flags[z]
                        row[f"z{z}"] = v
                    pending.append(row)
                    if len(pending) >= UPDATE_BATCH_SIZE:
                        flush_pending()

            print(
                f"  Batch finished in {time.monotonic() - t_batch:.2f}s "
                f"({hour_idx} hour(s) in range processed)",
                flush=True,
            )
            batch_start_ms = batch_end_ms

        flush_pending()
    finally:
        safe_close(session_gjk)
        safe_close(session_gbo)

    print("\nZone at-setpoint backfill finished.")


if __name__ == "__main__":
    ensure_zone_at_setpoint_columns()
    backfill_zone_at_setpoint()
    print("=" * 60 + " zone_at_setpoint complete " + "=" * 60)

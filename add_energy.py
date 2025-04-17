import os
import time
import dotenv
import pendulum
from sqlalchemy import create_engine, asc
from sqlalchemy.orm import sessionmaker
from gjk.models import MessageSql
from typing import List
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

dotenv.load_dotenv()

Base = declarative_base()

class HourlyElectricity(Base):
    __tablename__ = 'hourly_electricity'
    g_node_alias = Column(String, nullable=False, primary_key=True)
    short_alias = Column(String, nullable=False)
    hour_start_s = Column(BigInteger, nullable=False, primary_key=True)
    kwh = Column(Float, nullable=False)
    
    __table_args__ = (
        UniqueConstraint('hour_start_s', 'g_node_alias', name='hour_house_unique'),
    )

class EnergyDataset():
    def __init__(self, house_alias, start_ms, end_ms, timezone):
        engine = create_engine(os.getenv("GJK_DB_URL"))
        Session = sessionmaker(bind=engine)
        self.session = Session()
        engine_gbo = create_engine(os.getenv("GBO_DB_URL"))
        # Base.metadata.drop_all(engine_gbo)
        # Base.metadata.create_all(engine_gbo)
        SessionGbo = sessionmaker(bind=engine_gbo)
        self.session_gbo = SessionGbo()
        self.house_alias = house_alias
        self.dataset_file = f"energy_data_{self.house_alias}.csv"
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.timezone_str = timezone
        self.data_format = {
            'g_node_alias': [],
            'short_alias': [],
            'hour_start_ms': [],
            'kwh': [],
        }

    def generate_dataset(self):
        print("Generating dataset...")
        existing_dataset_dates = []
        if os.path.exists(self.dataset_file):
            print(f"Found existing dataset: {self.dataset_file}")
            df = pd.read_csv(self.dataset_file)
            existing_dataset_dates = list(df['hour_start_ms'])
        day_start_ms = int(pendulum.from_timestamp(self.start_ms/1000, tz=self.timezone_str).replace(hour=0, minute=0).timestamp()*1000)
        day_end_ms = day_start_ms + (23*60+7)*60*1000
        for day in range(200):
            if day_start_ms > self.end_ms or day_start_ms/1000 > time.time():
                if day_start_ms > self.end_ms:
                    print("day start ms is greater than end ms")
                print("\nDone.")
                return
            if day_start_ms in existing_dataset_dates and day_start_ms+(23*60)*60*1000 in existing_dataset_dates:
                print(f"\nAlready in dataset: {self.unix_ms_to_date(day_start_ms)}")
            else:
                self.add_data(day_start_ms, day_end_ms)
            day_start_ms += 24*3600*1000
            day_end_ms += 24*3600*1000

    def add_data(self, start_ms, end_ms):
        print(f"\nProcessing reports from: {self.unix_ms_to_date(start_ms)}")
        reports: List[MessageSql] = self.session.query(MessageSql).filter(
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            MessageSql.message_type_name == "report",
            MessageSql.message_persisted_ms >= start_ms,
            MessageSql.message_persisted_ms <= end_ms,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()
        
        print(f"Found {len(reports)} reports in database")
        if not reports:
            return
        
        formatted_data = pd.DataFrame(self.data_format)
        rows = []

        hour_start_ms = int(start_ms) - 3600*1000
        hour_end_ms = int(start_ms)

        while hour_end_ms <= end_ms:
            hour_start_ms += 3600*1000
            hour_end_ms += 3600*1000

            channels = {}
            for message in [
                m for m in reports
                if self.house_alias in m.from_alias
                and m.message_persisted_ms >= hour_start_ms - 7*60*1000
                and m.message_persisted_ms <= hour_end_ms + 7*60*1000
                ]:
                for channel in message.payload['ChannelReadingList']:
                    channel_name = channel['ChannelName']
                    if channel_name not in channels:
                        channels[channel_name] = {'times': [], 'values': []}
                    channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
                    channels[channel_name]['values'].extend(channel['ValueList'])
            if not channels:
                continue
            for channel in channels.keys():
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['values'] = list(sorted_values)
                channels[channel]['times'] = list(sorted_times)

            # HP heat out, electricity in, cop
            hp_channels = ['hp-idu-pwr', 'hp-odu-pwr']
            missing_channels = [c for c in hp_channels if c not in channels]
            if missing_channels: 
                print(missing_channels)
                continue

            timestep_seconds = 1
            num_points = int((hour_end_ms - hour_start_ms) / (timestep_seconds * 1000) + 1)

            csv_times = np.linspace(hour_start_ms, hour_end_ms, num_points)
            csv_times_dt = pd.to_datetime(csv_times, unit='ms', utc=True)
            csv_times_dt = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in csv_times_dt]

            csv_values = {}
            for channel in hp_channels:
                channels[channel]['times'] = pd.to_datetime(channels[channel]['times'], unit='ms', utc=True)
                channels[channel]['times'] = [x.tz_convert(self.timezone_str) for x in channels[channel]['times']]
                channels[channel]['times'] = [x.replace(tzinfo=None) for x in channels[channel]['times']]
                
                merged = pd.merge_asof(
                    pd.DataFrame({'times': csv_times_dt}),
                    pd.DataFrame(channels[channel]),
                    on='times',
                    direction='backward')
                csv_values[channel] = list(merged['values'])

            df = pd.DataFrame(csv_values)
            df['hp_power'] = df['hp-idu-pwr'] + df['hp-odu-pwr']
            hp_elec_in = round(float(np.mean(df['hp_power'])/1000),2)

            hour = str(len(formatted_data)) if len(formatted_data)>9 else '0'+str(len(formatted_data))
            print(f"{hour}:00 - HP: {hp_elec_in} kWh_e")

            row = [reports[0].from_alias, self.house_alias, hour_start_ms, hp_elec_in]
            formatted_data.loc[len(formatted_data)] = row 

            row = HourlyElectricity(
                g_node_alias=reports[0].from_alias,
                short_alias=self.house_alias,
                hour_start_s=int(hour_start_ms/1000),
                kwh=hp_elec_in
            )
            rows.append(row)
        
        try:
            self.session_gbo.add_all(rows)
            self.session_gbo.commit()
            print(f"Successfully inserted {len(rows)} new rows")
        except Exception as e:
            if 'hour_house_unique' in str(e) or "hourly_electricity_pkey" in str(e):  # Check if it's our unique constraint violation
                print("Some rows already exist in the database, filtering them out...")
                self.session_gbo.rollback()
                conflicting_rows = []
                for row in rows:
                    try:
                        self.session_gbo.add(row)
                        self.session_gbo.commit()
                    except Exception:
                        self.session_gbo.rollback()
                        conflicting_rows.append(row)
                # Filter out the conflicting rows
                rows = [row for row in rows if row not in conflicting_rows]
                # Insert the remaining rows
                if rows:
                    self.session_gbo.add_all(rows)
                    self.session_gbo.commit()
                    print(f"Successfully inserted {len(rows)} new rows")
                else:
                    print("All rows already existed in the database")
            else:
                self.session_gbo.rollback()
                raise Exception(f"Unexpected error: {e}")

        formatted_data.to_csv(
            self.dataset_file, 
            mode='a' if os.path.exists(self.dataset_file) else 'w',
            header=False if os.path.exists(self.dataset_file) else True, 
            index=False,
        )

    def unix_ms_to_date(self, time_ms):
        return str(pendulum.from_timestamp(time_ms/1000, tz=self.timezone_str).format('YYYY-MM-DD'))
    
    def to_fahrenheit(self, t):
        return round(t*9/5+32,1)
    
def generate(house_alias, start_year, start_month, start_day, end_year, end_month, end_day):
    timezone = 'America/New_York'
    start_ms = pendulum.datetime(start_year, start_month, start_day, tz=timezone).timestamp()*1000
    end_ms = pendulum.datetime(end_year, end_month, end_day, tz=timezone).timestamp()*1000
    s = EnergyDataset(house_alias, start_ms, end_ms, timezone)
    s.generate_dataset()

if __name__ == '__main__':
    generate(
        house_alias='fir', 
        start_year=2024, 
        start_month=12, 
        start_day=1,
        end_year=2025,
        end_month=5,
        end_day=31
    )
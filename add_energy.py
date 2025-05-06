import os
import time
import dotenv
import pendulum
from sqlalchemy import create_engine, asc, or_
from sqlalchemy.orm import sessionmaker
from gjk.models import MessageSql
from typing import List
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

dotenv.load_dotenv()

engine_gbo = create_engine(os.getenv("GBO_DB_URL"))
Base = declarative_base()

class HourlyElectricity(Base):
    __tablename__ = 'hourly_electricity'
    g_node_alias = Column(String, nullable=False, primary_key=True)
    short_alias = Column(String, nullable=False)
    hour_start_s = Column(BigInteger, nullable=False, primary_key=True)
    kwh = Column(Float, nullable=False)
    hp_kwh_th = Column(Float, nullable=True)
    storage_avg_temp_start_f = Column(Float, nullable=True)
    storage_avg_temp_end_f = Column(Float, nullable=True)
    buffer_avg_temp_start_f = Column(Float, nullable=True)
    buffer_avg_temp_end_f = Column(Float, nullable=True)
    relay_3_pulled_fraction = Column(Float, nullable=True)
    store_energy_in_flow_kwh = Column(Float, nullable=True)
    store_energy_in_avg_temp_kwh = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint('hour_start_s', 'g_node_alias', name='hour_house_unique'),
    )

# Drop existing tables
# Base.metadata.drop_all(engine_gbo)
# Base.metadata.create_all(engine_gbo)
# if os.path.exists(f"energy_data_beech.csv"):
#     os.remove(f"energy_data_beech.csv")

# SessionGbo = sessionmaker(bind=engine_gbo)
# session = SessionGbo()
# session.query(HourlyElectricity).filter(HourlyElectricity.short_alias == "fir").delete()
# session.commit()
# session.close()
# if os.path.exists(f"energy_data_fir.csv"):
#     os.remove(f"energy_data_fir.csv")

class EnergyDataset():
    def __init__(self, house_alias, start_ms, end_ms, timezone):
        engine = create_engine(os.getenv("GJK_DB_URL"))
        Session = sessionmaker(bind=engine)
        self.session = Session()
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
            'hp_kwh_th': [],
            'storage_avg_temp_start_f': [],
            'storage_avg_temp_end_f': [],
            'buffer_avg_temp_start_f': [],
            'buffer_avg_temp_end_f': [],
            'relay_3_pulled_fraction': [],
            'store_energy_in_flow_kwh': [],
            'store_energy_in_avg_temp_kwh': [],
        }

    def find_first_date(self):
        first_report: List[MessageSql] = self.session.query(MessageSql).filter(
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            or_(
                MessageSql.message_type_name == "batched.readings",
                MessageSql.message_type_name == "report"
            ),
            MessageSql.message_persisted_ms >= self.start_ms,
            MessageSql.message_persisted_ms <= self.end_ms,
        ).order_by(asc(MessageSql.message_persisted_ms)).first()
        self.start_ms = first_report.message_persisted_ms
        print(f"Data for {self.house_alias} starts at {self.unix_ms_to_date(self.start_ms)}")

    def generate_dataset(self):
        print("\nGenerating dataset...")
        self.find_first_date()
        existing_dataset_dates = []
        if os.path.exists(self.dataset_file):
            print(f"Found existing dataset: {self.dataset_file}")
            df = pd.read_csv(self.dataset_file)
            existing_dataset_dates = [int(x) for x in list(df['hour_start_ms'])]

        # Add data in batches of BATCH_SIZE hours
        BATCH_SIZE = 500
        batch_start_ms = int(pendulum.from_timestamp(self.start_ms/1000, tz=self.timezone_str).replace(hour=0, minute=0, microsecond=0).timestamp()*1000)
        batch_end_ms = batch_start_ms + BATCH_SIZE*3600*1000
        today_ms = int(time.time()*1000)
        
        while batch_start_ms < min(self.end_ms, today_ms):
            if existing_dataset_dates and int(batch_end_ms-3600*1000) <= max(existing_dataset_dates):
                print("Batch is already in data")
            else:
                self.add_data(batch_start_ms, batch_end_ms)
            batch_start_ms += BATCH_SIZE*3600*1000
            batch_end_ms += BATCH_SIZE*3600*1000

    def add_data(self, batch_start_ms, batch_end_ms):
        st = time.time()
        print(f"\nGathering reports from: {self.unix_ms_to_date(batch_start_ms)} to {self.unix_ms_to_date(batch_end_ms)}...")
        
        reports: List[MessageSql] = self.session.query(MessageSql).filter(
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            or_(
                MessageSql.message_type_name == "batched.readings",
                MessageSql.message_type_name == "report"
            ),
            MessageSql.message_persisted_ms >= batch_start_ms - 7*60*1000,
            MessageSql.message_persisted_ms <= batch_end_ms + 7*60*1000,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()
        
        print(f"Found {len(reports)} reports in database in {int(time.time()-st)} seconds")
        st = time.time()
        if not reports:
            return
        
        formatted_data = pd.DataFrame(self.data_format)
        rows = []

        hour_start_ms = int(batch_start_ms) - 3600*1000
        hour_end_ms = int(batch_start_ms)

        while hour_end_ms < batch_end_ms:
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
                    if message.message_type_name == 'report':
                        channel_name = channel['ChannelName']
                    elif message.message_type_name == 'batched.readings':
                        for dc in message.payload['DataChannelList']:
                            if dc['Id'] == channel['ChannelId']:
                                channel_name = dc['Name']
                    if channel_name not in channels:
                        channels[channel_name] = {'times': [], 'values': []}
                    channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
                    channels[channel_name]['values'].extend(channel['ValueList'])
            if not channels:
                print(f"No channels found in reports")
                continue
            for channel in channels.keys():
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['values'] = list(sorted_values)
                channels[channel]['times'] = list(sorted_times)

            # Heat pump
            hp_critical_channels = ['hp-idu-pwr', 'hp-odu-pwr']
            missing_channels = [c for c in hp_critical_channels if c not in channels]
            if missing_channels: 
                print(f"Missing critical HP data channels: {missing_channels}")
                continue

            additional_channels = [
                'hp-idu-pwr', 'hp-odu-pwr', 'hp-lwt', 'hp-ewt', 'primary-flow', 
                'store-flow', 'store-hot-pipe', 'store-cold-pipe', 'charge-discharge-relay3'
                ]
            hp_additional_channels = [x for x in additional_channels if 'hp' in x or 'primary' in x]
            store_additional_channels = [x for x in additional_channels if 'flow' in x or 'store' in x or 'charge' in x]

            timestep_seconds = 1
            num_points = int((hour_end_ms - hour_start_ms) / (timestep_seconds * 1000) + 1)

            csv_times = np.linspace(hour_start_ms, hour_end_ms, num_points)
            csv_times_dt = pd.to_datetime(csv_times, unit='ms', utc=True)
            csv_times_dt = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in csv_times_dt]

            csv_values = {}
            for channel in additional_channels:
                if channel not in channels or not channels[channel]['times']:
                    print(f"Missing channel data: {channel}")
                    continue
                channels[channel]['times'] = pd.to_datetime(channels[channel]['times'], unit='ms', utc=True)
                channels[channel]['times'] = [x.tz_convert(self.timezone_str) for x in channels[channel]['times']]
                channels[channel]['times'] = [x.replace(tzinfo=None) for x in channels[channel]['times']]
                
                try:
                    merged = pd.merge_asof(
                        pd.DataFrame({'times': csv_times_dt}),
                        pd.DataFrame(channels[channel]).ffill(),
                        on='times',
                        direction='backward'
                    )
                    csv_values[channel] = list(merged['values'])

                except Exception as e:
                    print(f"Error merging: {e}")
                    if sorted(csv_times_dt) != csv_times_dt:
                        print(f"\ncsv_times_dt: {csv_times_dt}\n")
                    elif sorted(channels[channel]['times']) != channels[channel]['times']:
                        print(f"\nchannels[channel]['times']: {channels[channel]['times']}\n")
                    merged = pd.merge_asof(
                        pd.DataFrame({'times': csv_times_dt}).sort_values('times'),
                        pd.DataFrame(channels[channel]).sort_values('times'),
                        on='times',
                        direction='backward'
                    )
                    csv_values[channel] = list(merged['values'])

            df = pd.DataFrame(csv_values)
            df['hp_power'] = df['hp-idu-pwr'] + df['hp-odu-pwr']
            hp_elec_in = round(float(np.mean(df['hp_power'])/1000),2)
            if not [c for c in hp_additional_channels if c not in csv_values]:
                # HP heat out
                df['lift_C'] = df['hp-lwt'] - df['hp-ewt']
                df['lift_C'] = [x/1000 if x>0 else 0 for x in list(df['lift_C'])]
                df['flow_kgs'] = df['primary-flow'] / 100 / 60 * 3.78541 
                df['heat_power_kW'] = [m*4187*lift/1000 for lift, m in zip(df['lift_C'], df['flow_kgs'])]
                df['cumulative_heat_kWh'] = df['heat_power_kW'].cumsum()
                df['cumulative_heat_kWh'] = df['cumulative_heat_kWh'] / 3600 * timestep_seconds
                hp_heat_out = round(list(df['cumulative_heat_kWh'])[-1] - list(df['cumulative_heat_kWh'])[0],2)
                if np.isnan(hp_heat_out):
                    hp_heat_out = 0
            else:
                hp_heat_out = 0 if hp_elec_in < 0.5 else None

            if not [c for c in store_additional_channels if c not in csv_values]:
                # Relay 3 pulled fraction
                df['relay3_cumulative'] = df['charge-discharge-relay3'].cumsum()
                relay3_pulled_fraction = round(list(df['relay3_cumulative'])[-1] / len(df['relay3_cumulative']), 2)
                # Store energy change
                df['store_lift_C'] = np.where(
                    df['charge-discharge-relay3'] == 0,
                    df['store-hot-pipe'] - df['store-cold-pipe'],
                    df['store-cold-pipe'] - df['store-hot-pipe']
                )
                df['store_lift_C'] = df['store_lift_C']/1000
                df['store_flow_kgs'] = np.where(
                    df['charge-discharge-relay3'] == 0,
                    df['store-flow'] / 100 / 60 * 3.78541,
                    df['primary-flow'] / 100 / 60 * 3.78541
                )
                df['store_heat_power_kW'] = [m*4187*lift/1000 for lift, m in zip(df['store_lift_C'], df['store_flow_kgs'])]
                df['store_cumulative_heat_kWh'] = df['store_heat_power_kW'].cumsum()
                df['store_cumulative_heat_kWh'] = df['store_cumulative_heat_kWh'] / 3600 * timestep_seconds
                store_heat_in_flow = -round(list(df['store_cumulative_heat_kWh'])[-1] - list(df['store_cumulative_heat_kWh'])[0],2)
            else:
                store_heat_in_flow = None
                relay3_pulled_fraction = None

            # Buffer
            buffer_channels = [x for x in channels if 'buffer' in x and 'depth' in x and 'micro' not in x]
            hour_start_times, hour_start_values = [], []
            hour_end_times, hour_end_values = [], []

            for channel in buffer_channels:
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['times'] = list(sorted_times)      
                channels[channel]['values'] = list(sorted_values)

                times_from_start = [abs(time-hour_start_ms) for time in channels[channel]['times']]
                closest_index = times_from_start.index(min(times_from_start))
                hour_start_times.append(channels[channel]['times'][closest_index])
                hour_start_values.append(channels[channel]['values'][closest_index]/1000)

                times_from_end = [abs(time-hour_end_ms) for time in channels[channel]['times']]
                closest_index = times_from_end.index(min(times_from_end))
                hour_end_times.append(channels[channel]['times'][closest_index])
                hour_end_values.append(channels[channel]['values'][closest_index]/1000)

            if not hour_start_values or not hour_end_values or hour_end_times[-1] - hour_start_times[-1] < 45*60*1000:
                average_buffer_temp_start = None
                average_buffer_temp_end = None
            else:
                average_buffer_temp_start = self.to_fahrenheit(sum(hour_start_values)/len(hour_start_values))
                average_buffer_temp_end = self.to_fahrenheit(sum(hour_end_values)/len(hour_end_values))

            # Storage
            storage_channels = [x for x in channels if 'tank' in x and 'depth' in x and 'micro' not in x]
            hour_start_times, hour_start_values = [], []
            hour_end_times, hour_end_values = [], []

            for channel in storage_channels:
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['times'] = list(sorted_times)      
                channels[channel]['values'] = list(sorted_values)

                times_from_start = [abs(time-hour_start_ms) for time in channels[channel]['times']]
                closest_index = times_from_start.index(min(times_from_start))
                hour_start_times.append(channels[channel]['times'][closest_index])
                hour_start_values.append(channels[channel]['values'][closest_index]/1000)

                times_from_end = [abs(time-hour_end_ms) for time in channels[channel]['times']]
                closest_index = times_from_end.index(min(times_from_end))
                hour_end_times.append(channels[channel]['times'][closest_index])
                hour_end_values.append(channels[channel]['values'][closest_index]/1000)

            if not hour_start_values or not hour_end_values or hour_end_times[-1] - hour_start_times[-1] < 45*60*1000:
                average_store_temp_start = None
                average_store_temp_end = None
                store_energy_in_avg_temp_kwh = None
            else:
                avg_store_temp_start = sum(hour_start_values)/len(hour_start_values)
                avg_store_temp_end = sum(hour_end_values)/len(hour_end_values)
                average_store_temp_start = self.to_fahrenheit(avg_store_temp_start)
                average_store_temp_end = self.to_fahrenheit(avg_store_temp_end)
                store_energy_in_avg_temp_kwh = round(3*120*3.785*4.187/3600*(avg_store_temp_end-avg_store_temp_start),2)

            print(f"{self.unix_ms_to_date(hour_start_ms)} - HP: {hp_elec_in} kWh_e, {hp_heat_out} kWh_th")

            row = [
                reports[0].from_alias, 
                self.house_alias, 
                hour_start_ms, 
                hp_elec_in, 
                hp_heat_out, 
                average_store_temp_start, 
                average_store_temp_end, 
                average_buffer_temp_start, 
                average_buffer_temp_end,
                relay3_pulled_fraction,
                store_heat_in_flow,
                store_energy_in_avg_temp_kwh
            ]
            row = [x if x is not None else np.nan for x in row]
            formatted_data.loc[len(formatted_data)] = row 

            row = HourlyElectricity(
                g_node_alias=reports[0].from_alias,
                short_alias=self.house_alias,
                hour_start_s=int(hour_start_ms/1000),
                kwh=hp_elec_in,
                hp_kwh_th=hp_heat_out,
                storage_avg_temp_start_f=average_store_temp_start,
                storage_avg_temp_end_f=average_store_temp_end,
                buffer_avg_temp_start_f=average_buffer_temp_start,
                buffer_avg_temp_end_f=average_buffer_temp_end,
                relay_3_pulled_fraction=relay3_pulled_fraction,
                store_energy_in_flow_kwh=store_heat_in_flow,
                store_energy_in_avg_temp_kwh=store_energy_in_avg_temp_kwh,
            )
            rows.append(row)
        
        try:
            self.session_gbo.add_all(rows)
            self.session_gbo.commit()
            print(f"Successfully inserted {len(rows)} new rows in {int(time.time()-st)} seconds")
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

        formatted_data['datetime_str'] = formatted_data['hour_start_ms'].apply(self.unix_ms_to_date)
        formatted_data.to_csv(
            self.dataset_file, 
            mode='a' if os.path.exists(self.dataset_file) else 'w',
            header=False if os.path.exists(self.dataset_file) else True, 
            index=False,
        )

    def unix_ms_to_date(self, time_ms):
        return str(pendulum.from_timestamp(time_ms/1000, tz=self.timezone_str).format('YYYY-MM-DD HH:mm'))
    
    def to_fahrenheit(self, t):
        return round(t*9/5+32,1)
    
def generate(house_alias, start_year, start_month, start_day, end_year, end_month, end_day):
    timezone = 'America/New_York'
    start_ms = pendulum.datetime(start_year, start_month, start_day, tz=timezone).timestamp()*1000
    end_ms = pendulum.datetime(end_year, end_month, end_day, tz=timezone).timestamp()*1000
    s = EnergyDataset(house_alias, start_ms, end_ms, timezone)
    s.generate_dataset()

if __name__ == '__main__':
    houses_to_generate = ['fir', 'maple', 'elm']
    for house in houses_to_generate:
        generate(
            house_alias=house, 
            start_year=2024, 
            start_month=9, 
            start_day=1,
            end_year=2025,
            end_month=5,
            end_day=31
        )

import os
import time
import dotenv
import pendulum
from sqlalchemy import create_engine, asc, or_, and_
from sqlalchemy.orm import sessionmaker
from gjk.models import MessageSql
from typing import List
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, Column, String, Float, BigInteger, UniqueConstraint, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import matplotlib.pyplot as plt
from add_bid_column import AtnBid, extract_pq_pairs

DROP_EXISTING_DATA = False

dotenv.load_dotenv()
gbo_db_url = os.getenv("GBO_DB_URL")
engine_gbo = create_engine(gbo_db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://"))
Base = declarative_base()

class HourlyElectricity(Base):
    __tablename__ = 'hourly_electricity'
    g_node_alias = Column(String, nullable=False, primary_key=True)
    short_alias = Column(String, nullable=False)
    hour_start_s = Column(BigInteger, nullable=False, primary_key=True)

    hp_kwh_el = Column(Float, nullable=False)
    hp_kwh_th = Column(Float, nullable=True)
    dist_kwh = Column(Float, nullable=True)
    store_change_kwh = Column(Float, nullable=True)

    hp_avg_lwt = Column(Float, nullable=True)
    hp_avg_ewt = Column(Float, nullable=True)
    dist_avg_swt = Column(Float, nullable=True)
    dist_avg_rwt = Column(Float, nullable=True)

    buffer_depth1_start = Column(Float, nullable=True)
    buffer_depth2_start = Column(Float, nullable=True)
    buffer_depth3_start = Column(Float, nullable=True)
    buffer_depth4_start = Column(Float, nullable=True)
    tank1_depth1_start = Column(Float, nullable=True)
    tank1_depth2_start = Column(Float, nullable=True)
    tank1_depth3_start = Column(Float, nullable=True)
    tank1_depth4_start = Column(Float, nullable=True)
    tank2_depth1_start = Column(Float, nullable=True)
    tank2_depth2_start = Column(Float, nullable=True)
    tank2_depth3_start = Column(Float, nullable=True)
    tank2_depth4_start = Column(Float, nullable=True)
    tank3_depth1_start = Column(Float, nullable=True)
    tank3_depth2_start = Column(Float, nullable=True)
    tank3_depth3_start = Column(Float, nullable=True)
    tank3_depth4_start = Column(Float, nullable=True)

    relay_3_pulled_fraction = Column(Float, nullable=True)
    relay_5_pulled_fraction = Column(Float, nullable=True)
    relay_6_pulled_fraction = Column(Float, nullable=True)
    relay_9_pulled_fraction = Column(Float, nullable=True)

    zone1_heatcall_fraction = Column(Float, nullable=True)
    zone2_heatcall_fraction = Column(Float, nullable=True)
    zone3_heatcall_fraction = Column(Float, nullable=True)
    zone4_heatcall_fraction = Column(Float, nullable=True)

    oat_f = Column(Float, nullable=True)
    ws_mph = Column(Float, nullable=True)
    total_usd_per_mwh = Column(Float, nullable=True)
    flo = Column(Boolean, nullable=True)

    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    intermediate_power_kw = Column(Float, nullable=True)
    intermediate_rswt = Column(Float, nullable=True)
    dd_power_kw = Column(Float, nullable=True)
    dd_rswt = Column(Float, nullable=True)
    dd_delta_t = Column(Float, nullable=True)
    bid = Column(String, nullable=True)
    
    __table_args__ = (
        UniqueConstraint('hour_start_s', 'g_node_alias', name='hour_house_unique'),
    )

if DROP_EXISTING_DATA:
    print("\nWARNING: Continuing will drop existing data")
    continue_dropping = input("Continue? (y/n): ")
    if continue_dropping != 'y':
        print("Exiting...\n")
        exit()
    Base.metadata.drop_all(engine_gbo)
    print("Existing data dropped")
    Base.metadata.create_all(engine_gbo)
    if os.path.exists(f"energy_data_beech.csv"):
        os.remove(f"energy_data_beech.csv")


class HourlyData:
    def __init__(self, house_alias:str, start_ms:int, end_ms:int, timezone:str, find_first_date:bool=False):
        self.house_alias = house_alias
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.timezone_str = timezone

        self.initialize_sqlalchemy_sessions()
        self.initialize_parameters()
        if find_first_date:
            self.find_first_date()
        
    def initialize_sqlalchemy_sessions(self):
        # Journaldb
        gjk_db_url = os.getenv("GJK_DB_URL").replace("asyncpg", "psycopg")
        engine_gjk = create_engine(gjk_db_url)
        Session = sessionmaker(bind=engine_gjk)
        self.session_gjk = Session()

        # Backofficedb
        gbo_db_url = os.getenv("GBO_DB_URL").replace("asyncpg", "psycopg")
        engine_gbo = create_engine(gbo_db_url)
        SessionGbo = sessionmaker(bind=engine_gbo)
        self.session_gbo = SessionGbo()

        # Data format
        self.data_format = {col.name: [] for col in HourlyElectricity.__table__.columns}

    def initialize_parameters(self):
        # Whitewire threshold
        whitewire_threshold_watts = {'beech': 100, 'elm': 1, 'default': 20}
        if self.house_alias in whitewire_threshold_watts:
            self.whitewire_threshold = whitewire_threshold_watts[self.house_alias]
        else:
            self.whitewire_threshold = whitewire_threshold_watts['default']

        # Primary pump threshold
        primary_pump_gpm = {'beech': 5.5, 'default': 5}
        if self.house_alias in primary_pump_gpm:
            self.primary_pump_gpm = primary_pump_gpm[self.house_alias]
        else:
            self.primary_pump_gpm = primary_pump_gpm['default']

        # Store pump threshold
        store_pump_gpm = {'beech': 1.5, 'default': 1.5}
        if self.house_alias in store_pump_gpm:
            self.store_pump_gpm = store_pump_gpm[self.house_alias]
        else:
            self.store_pump_gpm = store_pump_gpm['default']

        self.stop_flow_processing_time_ms = pendulum.datetime(2025,1,1,tz=self.timezone_str).timestamp()*1000
        self.small_batch_size = 20
        self.large_batch_size = 100

    def find_first_date(self):
        first_report: List[MessageSql] = self.session_gjk.query(MessageSql).filter(
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            or_(
                MessageSql.message_type_name == "batched.readings",
                MessageSql.message_type_name == "report"
            ),
            MessageSql.message_created_ms >= self.start_ms,
            MessageSql.message_created_ms <= self.end_ms,
        ).order_by(asc(MessageSql.message_created_ms)).first()
        self.start_ms = first_report.message_created_ms
        print(f"Data for {self.house_alias} starts at {self.unix_ms_to_date(self.start_ms)}")

    def generate_hourly_data(self):
        print("\n" + "="*100 + f"\nGenerating hourly data for {self.house_alias.capitalize()}" + "\n" + "="*100)
        batch_start_ms = int(pendulum.from_timestamp(self.start_ms/1000, tz=self.timezone_str).replace(hour=0, minute=0, microsecond=0).timestamp()*1000)
        batch_size = self.small_batch_size if batch_start_ms < self.stop_flow_processing_time_ms else self.large_batch_size
        batch_end_ms = batch_start_ms + batch_size*3600*1000
        now_ms = int(time.time()*1000)
        
        while batch_start_ms < min(self.end_ms, now_ms):
            batch_end_ms = min(batch_end_ms, self.end_ms, now_ms)            
            try:
                self.add_batch(batch_start_ms, batch_end_ms)
            except Exception as e:
                print(f"Error adding data from {self.unix_ms_to_date(batch_start_ms)} to {self.unix_ms_to_date(batch_end_ms)}: {e}")
            batch_size = self.small_batch_size if batch_start_ms < self.stop_flow_processing_time_ms else self.large_batch_size
            batch_start_ms += batch_size*3600*1000
            batch_end_ms += batch_size*3600*1000

    def add_batch(self, batch_start_ms, batch_end_ms):
        print(f"\nBatch from {self.unix_ms_to_date(batch_start_ms)} to {self.unix_ms_to_date(batch_end_ms)}")
        messages: List[MessageSql] = self.session_gjk.query(MessageSql).filter(
            or_(
                and_(
                    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{self.house_alias}.scada",
                    or_(
                        MessageSql.message_type_name == "batched.readings",
                        MessageSql.message_type_name == "report",
                        MessageSql.message_type_name == "layout.lite"
                    )
                ),
                and_(
                    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{self.house_alias}",
                    MessageSql.message_type_name == "atn.bid"
                ),
                MessageSql.message_type_name == "flo.params.house0",
            ),
            MessageSql.message_created_ms >= batch_start_ms - 7*60*1000,
            MessageSql.message_created_ms <= batch_end_ms + 7*60*1000,
        ).order_by(asc(MessageSql.message_created_ms)).all()

        reports = [m for m in messages if m.message_type_name in ['report', 'batched.readings']]
        flo_params = [m for m in messages if m.message_type_name == 'flo.params.house0']
        atn_bids = [m for m in messages if m.message_type_name == 'atn.bid']
        layout_lites = [m for m in messages if m.message_type_name == 'layout.lite']
        
        if not reports:
            print(f"No reports found in batch.")
            return
        
        # Generate a row per hour in the batch
        batch_rows = []
        hour_start_ms = int(batch_start_ms) - 3600*1000
        hour_end_ms = int(batch_start_ms)

        while hour_end_ms < batch_end_ms:
            hour_start_ms += 3600*1000
            hour_end_ms += 3600*1000

            selected_messages = [
                m for m in reports
                if m.message_created_ms >= hour_start_ms-7*60*1000 
                and m.message_created_ms <= hour_end_ms+7*60*1000
                ]

            # Organize data by channel name
            channels = {}
            for message in selected_messages:
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
                continue
            for channel in channels.keys():
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['values'] = list(sorted_values)
                channels[channel]['times'] = list(sorted_times)

            # Get synchronous data for required data channels
            required_channels = [
                'hp-idu-pwr', 'hp-odu-pwr', 'hp-lwt', 'hp-ewt', 'primary-flow', 'primary-pump-pwr',
                'store-flow', 'store-hot-pipe', 'store-cold-pipe', 'store-pump-pwr',
                'charge-discharge-relay3', 'hp-failsafe-relay5', 'hp-scada-ops-relay6', 'store-pump-failsafe-relay9',
                'dist-swt', 'dist-rwt', 'dist-flow'
            ] + [
                x for x in channels if 'zone' in x and 'whitewire' in x
            ]
            hp_required_channels = [x for x in required_channels if 'hp' in x or 'primary-flow' in x]
            store_required_channels = [x for x in required_channels if 'flow' in x or 'store' in x or 'relay3' in x and 'pwr' not in x]
            dist_required_channels = [x for x in required_channels if 'dist' in x]

            timestep_seconds = 1
            num_points = int((hour_end_ms - hour_start_ms) / (timestep_seconds * 1000) + 1)
            sync_times = np.linspace(hour_start_ms, hour_end_ms, num_points)
            sync_times_dt = pd.to_datetime(sync_times, unit='ms', utc=True)
            sync_times_dt = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in sync_times_dt]

            sync_values = {'times': sync_times}
            for channel in required_channels:
                if channel not in channels or not channels[channel]['times']:
                    continue
                channels[channel]['times'] = pd.to_datetime(channels[channel]['times'], unit='ms', utc=True)
                channels[channel]['times'] = [x.tz_convert(self.timezone_str) for x in channels[channel]['times']]
                channels[channel]['times'] = [x.replace(tzinfo=None) for x in channels[channel]['times']]
                try:
                    merged = pd.merge_asof(
                        pd.DataFrame({'times': sync_times_dt}),
                        pd.DataFrame(channels[channel]).ffill(),
                        on='times',
                        direction='backward'
                    )
                    sync_values[channel] = list(merged['values'])
                except Exception as e:
                    print(f"Error merging {channel} data around {sync_times_dt[0]}: {e}")

            # Calculations from synchronous data
            df = pd.DataFrame(sync_values)

            # Initialize variables to None
            hp_elec_in = None
            hp_heat_out = None
            dist_kwh = None
            store_change_kwh = None
            hp_avg_lwt = None
            hp_avg_ewt = None
            dist_avg_swt = None
            dist_avg_rwt = None
            relay3_pulled_fraction = None
            relay5_pulled_fraction = None
            relay6_pulled_fraction = None
            relay9_pulled_fraction = None
            zone1_heatcall_fraction = None
            zone2_heatcall_fraction = None
            zone3_heatcall_fraction = None
            zone4_heatcall_fraction = None
            oat_f = None
            ws_mph = None
            total_usd_per_mwh = None
            alpha = None
            beta = None
            gamma = None
            intermediate_power_kw = None
            intermediate_rswt = None
            dd_power_kw = None
            dd_rswt = None
            dd_delta_t = None
            bid = None

            # ------------------------------------------------------------------------------------------------
            # Heat pump: hp_elec_in, hp_heat_out, hp_avg_lwt, hp_avg_ewt
            # ------------------------------------------------------------------------------------------------

            if 'hp-idu-pwr' in sync_values and 'hp-odu-pwr' in sync_values:
                df['hp-elec-in'] = df['hp-idu-pwr'] + df['hp-odu-pwr']
                hp_elec_in = round(float(np.mean(df['hp-elec-in'])/1000),2)
            else:
                continue

            if not [c for c in hp_required_channels if c not in sync_values] or (
                'primary-pump-pwr' in sync_values 
                and not [c for c in hp_required_channels if c not in sync_values and 'primary-flow' not in c]
            ):
                # Process primary flow
                if 'primary-pump-pwr' in sync_values and hour_end_ms < self.stop_flow_processing_time_ms or 'primary-flow' not in sync_values:
                    primary_flow_processed = []
                    last_correct_gpm = self.primary_pump_gpm
                    if 'primary-flow' in sync_values:
                        last_pump_off_ms = None
                        for i in range(len(df)):
                            value_gpm = float(df['primary-flow'][i]/100)
                            value_watts = float(df['primary-pump-pwr'][i])
                            pump_on = value_watts > 10
                            if not pump_on:
                                last_pump_off_ms = df['times'][i]
                            # Check if the pump was recently turned on
                            pump_just_turned_on = False
                            if last_pump_off_ms and pump_on and df['times'][i] - last_pump_off_ms < 20*1000:
                                pump_just_turned_on = True
                            # Check if the pump will soon be turned off and flow has stopped
                            pump_about_to_be_turned_off = False
                            min_pump_pwr_in_next_seconds = min(df[(df['times']>=df['times'][i]) & (df['times']<=df['times'][i]+10*1000)]['primary-pump-pwr'])
                            max_flow_in_next_seconds = max(df[(df['times']>=df['times'][i]) & (df['times']<=df['times'][i]+10*1000)]['primary-flow'])
                            if min_pump_pwr_in_next_seconds < 10 and max_flow_in_next_seconds < self.primary_pump_gpm-2:
                                pump_about_to_be_turned_off = True
                            # Update last correct GPM                            
                            if not np.isnan(value_gpm):
                                if pump_on and value_gpm > self.primary_pump_gpm-1:
                                    last_correct_gpm = value_gpm
                                elif (pump_on and value_gpm < self.primary_pump_gpm-1 
                                    and not pump_just_turned_on and not pump_about_to_be_turned_off):
                                    value_gpm = last_correct_gpm
                            else:
                                if pump_on and not pump_just_turned_on and not pump_about_to_be_turned_off:
                                    value_gpm = last_correct_gpm
                                else:
                                    value_gpm = 0
                            primary_flow_processed.append(value_gpm*100)
                    else:
                        for i in range(len(df)):
                            value_watts = float(df['primary-pump-pwr'][i])
                            pump_on = value_watts > 10
                            if pump_on:
                                value_gpm = last_correct_gpm
                            else:
                                value_gpm = 0
                            primary_flow_processed.append(value_gpm*100)
                    df['primary-flow'] = primary_flow_processed

                # Compute heat pump heat output
                df['lift_C'] = df['hp-lwt'] - df['hp-ewt']
                df['lift_C'] = df['lift_C']/1000
                df['flow_kgs'] = df['primary-flow'] / 100 / 60 * 3.78541 
                df['heat_power_kW'] = [m*4187*lift/1000 for lift, m in zip(df['lift_C'], df['flow_kgs'])]
                df['cumulative_heat_kWh'] = df['heat_power_kW'].cumsum()
                df['cumulative_heat_kWh'] = df['cumulative_heat_kWh'] / 3600 * timestep_seconds
                non_nan_cumulative_heat = [x for x in df['cumulative_heat_kWh'] if not np.isnan(x)]
                if non_nan_cumulative_heat:
                    first_non_nan_cumulative_heat = non_nan_cumulative_heat[0]
                    last_non_nan_cumulative_heat = non_nan_cumulative_heat[-1]
                    hp_heat_out = round(last_non_nan_cumulative_heat - first_non_nan_cumulative_heat,2)
                
                # Heat pump average LWT and EWT
                hp_avg_lwt = self.to_fahrenheit(float(np.mean(df['hp-lwt'])/1000))
                hp_avg_ewt = self.to_fahrenheit(float(np.mean(df['hp-ewt'])/1000))

            # ------------------------------------------------------------------------------------------------
            # Distribution: dist_kwh, dist_avg_swt, dist_avg_rwt
            # ------------------------------------------------------------------------------------------------
            
            if not [c for c in dist_required_channels if c not in sync_values]:
                df['dist_flow_kgs'] = df['dist-flow'] / 100 / 60 * 3.78541
                df['dist_lift_C'] = df['dist-swt'] - df['dist-rwt']
                df['dist_lift_C'] = df['dist_lift_C']/1000
                df['dist_heat_power_kW'] = [m*4187*lift/1000 for lift, m in zip(df['dist_lift_C'], df['dist_flow_kgs'])]
                df['dist_cumulative_heat_kWh'] = df['dist_heat_power_kW'].cumsum()
                df['dist_cumulative_heat_kWh'] = df['dist_cumulative_heat_kWh'] / 3600 * timestep_seconds
                dist_kwh = round(list(df['dist_cumulative_heat_kWh'])[-1] - list(df['dist_cumulative_heat_kWh'])[0],2)   
                dist_avg_swt = self.to_fahrenheit(float(np.mean(df['dist-swt'])/1000))
                dist_avg_rwt = self.to_fahrenheit(float(np.mean(df['dist-rwt'])/1000))

            # ------------------------------------------------------------------------------------------------
            # Storage: store_change_kwh
            # ------------------------------------------------------------------------------------------------

            if not [c for c in store_required_channels if c not in sync_values] or (
                'store-pump-pwr' in sync_values and 'primary-flow' in df 
                and not [c for c in store_required_channels if c not in sync_values and 'store-flow' not in c and 'primary-flow' not in c]
            ):
                # Process store flow
                if ('store-pump-pwr' in sync_values and hour_end_ms < self.stop_flow_processing_time_ms) or 'store-flow' not in sync_values:
                    store_flow_processed = []
                    last_correct_gpm = self.store_pump_gpm
                    if 'store-flow' in sync_values:
                        last_pump_off_ms = None
                        for i in range(len(df)):
                            value_gpm = float(df['store-flow'][i]/100)
                            value_watts = float(df['store-pump-pwr'][i])
                            pump_on = value_watts > 5
                            if not pump_on:
                                last_pump_off_ms = df['times'][i]

                            # Check if the pump was recently turned on
                            pump_just_turned_on = False
                            if last_pump_off_ms and pump_on and df['times'][i] - last_pump_off_ms < 20*1000:
                                pump_just_turned_on = True

                            # Check if the pump will soon be turned off and flow has stopped
                            pump_about_to_be_turned_off = False
                            min_pump_pwr_in_next_seconds = min(df[(df['times']>=df['times'][i]) & (df['times']<=df['times'][i]+10*1000)]['store-pump-pwr'])
                            max_flow_in_next_seconds = max(df[(df['times']>=df['times'][i]) & (df['times']<=df['times'][i]+10*1000)]['store-flow'])
                            if min_pump_pwr_in_next_seconds < 5 and max_flow_in_next_seconds < self.store_pump_gpm-2:
                                pump_about_to_be_turned_off = True

                            # Update last correct GPM                            
                            if not np.isnan(value_gpm):
                                if pump_on and value_gpm > self.store_pump_gpm-1:
                                    last_correct_gpm = value_gpm
                                elif (pump_on and value_gpm < self.store_pump_gpm-1 
                                    and not pump_just_turned_on and not pump_about_to_be_turned_off):
                                    value_gpm = last_correct_gpm
                            else:
                                if pump_on and not pump_just_turned_on and not pump_about_to_be_turned_off:
                                    value_gpm = last_correct_gpm
                                else:
                                    value_gpm = 0

                            store_flow_processed.append(value_gpm*100)
                    else:
                        for i in range(len(df)):
                            value_watts = float(df['store-pump-pwr'][i])
                            pump_on = value_watts > 5
                            if pump_on:
                                value_gpm = last_correct_gpm
                            else:
                                value_gpm = 0
                            store_flow_processed.append(value_gpm*100)
                    df['store-flow'] = store_flow_processed

                # Compute storage heat output
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
                store_change_kwh = -round(list(df['store_cumulative_heat_kWh'])[-1] - list(df['store_cumulative_heat_kWh'])[0],2)

                # Pipe energy
                df['pipe_lift_C'] = np.where(
                    df['charge-discharge-relay3'] == 0,
                    0,
                    (df['hp-lwt']-df['store-hot-pipe']) + (df['store-cold-pipe']-df['hp-ewt'])
                )
                df['pipe_lift_C'] = df['pipe_lift_C']/1000
                df['pipe_flow_kgs'] = np.where(
                    df['charge-discharge-relay3'] == 0,
                    df['store-flow'] / 100 / 60 * 3.78541,
                    df['primary-flow'] / 100 / 60 * 3.78541
                )
                df['pipe_heat_power_kW'] = [m*4187*lift/1000 for lift, m in zip(df['pipe_lift_C'], df['pipe_flow_kgs'])]
                df['pipe_cumulative_heat_kWh'] = df['pipe_heat_power_kW'].cumsum()
                df['pipe_cumulative_heat_kWh'] = df['pipe_cumulative_heat_kWh'] / 3600 * timestep_seconds

            # ------------------------------------------------------------------------------------------------
            # Tank temperatures: buffer_temps, storage_temps
            # ------------------------------------------------------------------------------------------------
            
            # Buffer temperatures
            buffer_channels = ['buffer-depth1', 'buffer-depth2', 'buffer-depth3', 'buffer-depth4']
            buffer_temps = {x: None for x in buffer_channels}
            for channel in [x for x in buffer_channels if x in channels]:
                times_from_start = [abs(time-hour_start_ms) for time in channels[channel]['times']]
                closest_index = times_from_start.index(min(times_from_start))
                buffer_temps[channel] = self.to_fahrenheit(channels[channel]['values'][closest_index]/1000)

            # Storage temperatures
            storage_channels = [
                'tank1-depth1', 'tank1-depth2', 'tank1-depth3', 'tank1-depth4', 
                'tank2-depth1', 'tank2-depth2', 'tank2-depth3', 'tank2-depth4', 
                'tank3-depth1', 'tank3-depth2', 'tank3-depth3', 'tank3-depth4'
            ]
            storage_temps = {x: None for x in storage_channels}
            for channel in [x for x in storage_channels if x in channels]:
                times_from_start = [abs(time-hour_start_ms) for time in channels[channel]['times']]
                closest_index = times_from_start.index(min(times_from_start))
                storage_temps[channel] = self.to_fahrenheit(channels[channel]['values'][closest_index]/1000)

            # ------------------------------------------------------------------------------------------------
            # Relays and zones
            # ------------------------------------------------------------------------------------------------

            # Relays
            if [c for c in sync_values if 'relay3' in c]:
                df['relay3_cumulative'] = df['charge-discharge-relay3'].cumsum()
                relay3_pulled_fraction = round(list(df['relay3_cumulative'])[-1] / len(df['relay3_cumulative']), 2)
            if [c for c in sync_values if 'relay5' in c]:
                df['relay5_cumulative'] = df['hp-failsafe-relay5'].cumsum()
                relay5_pulled_fraction = round(list(df['relay5_cumulative'])[-1] / len(df['relay5_cumulative']), 2)
            if [c for c in sync_values if 'relay6' in c]:
                df['relay6_cumulative'] = df['hp-scada-ops-relay6'].cumsum()
                relay6_pulled_fraction = round(list(df['relay6_cumulative'])[-1] / len(df['relay6_cumulative']), 2)
            if [c for c in sync_values if 'relay9' in c]:
                df['relay9_cumulative'] = df['store-pump-failsafe-relay9'].cumsum()
                relay9_pulled_fraction = round(list(df['relay9_cumulative'])[-1] / len(df['relay9_cumulative']), 2)

            # Zones
            whitewire_channels = [c for c in sync_values if 'whitewire' in c]
            if [c for c in whitewire_channels if 'zone1' in c]:
                zone1_ch = [c for c in whitewire_channels if 'zone1' in c][0]
                df[zone1_ch] = [int(abs(x)>self.whitewire_threshold) for x in df[zone1_ch]]
                df['zone1_cumulative'] = df[zone1_ch].cumsum()
                zone1_heatcall_fraction = round(list(df['zone1_cumulative'])[-1] / len(df['zone1_cumulative']), 2)
            if [c for c in whitewire_channels if 'zone2' in c]:
                zone2_ch = [c for c in whitewire_channels if 'zone2' in c][0]
                df[zone2_ch] = [int(abs(x)>self.whitewire_threshold) for x in df[zone2_ch]]
                df['zone2_cumulative'] = df[zone2_ch].cumsum()
                zone2_heatcall_fraction = round(list(df['zone2_cumulative'])[-1] / len(df['zone2_cumulative']), 2)
            if [c for c in whitewire_channels if 'zone3' in c]:
                zone3_ch = [c for c in whitewire_channels if 'zone3' in c][0]
                df[zone3_ch] = [int(abs(x)>self.whitewire_threshold) for x in df[zone3_ch]]
                df['zone3_cumulative'] = df[zone3_ch].cumsum()
                zone3_heatcall_fraction = round(list(df['zone3_cumulative'])[-1] / len(df['zone3_cumulative']), 2)
            if [c for c in whitewire_channels if 'zone4' in c]:
                zone4_ch = [c for c in whitewire_channels if 'zone4' in c][0]
                df[zone4_ch] = [int(abs(x)>self.whitewire_threshold) for x in df[zone4_ch]]
                df['zone4_cumulative'] = df[zone4_ch].cumsum()
                zone4_heatcall_fraction = round(list(df['zone4_cumulative'])[-1] / len(df['zone4_cumulative']), 2)

            # ------------------------------------------------------------------------------------------------
            # Advanced: weather, prices, house parameters, bids
            # ------------------------------------------------------------------------------------------------

            # Get weather and price data
            for f in flo_params:
                if f.payload['StartUnixS'] == hour_start_ms/1000:
                    oat_f = f.payload['OatForecastF'][0]
                    ws_mph = f.payload['WindSpeedForecastMph'][0]
                    total_usd_per_mwh = f.payload['LmpForecast'][0] + f.payload['DistPriceForecast'][0]
                    total_usd_per_mwh = round(total_usd_per_mwh, 3)
                    break

            # Determine if FLO or HomeAlone and get house parameters
            flo_tf = False
            for f in flo_params:
                if f.payload['StartUnixS'] == hour_start_ms/1000: 
                    if f.from_alias == f"hw1.isone.me.versant.keene.{self.house_alias}":
                        flo_tf = True
                        alpha = f.payload['AlphaTimes10']/10
                        beta = f.payload['BetaTimes100']/100
                        gamma = f.payload['GammaEx6']/10**6
                        intermediate_power_kw = f.payload['IntermediatePowerKw']
                        intermediate_rswt = f.payload['IntermediateRswtF']
                        dd_power_kw = f.payload['DdPowerKw']
                        dd_rswt = f.payload['DdRswtF']
                        dd_delta_t = f.payload['DdDeltaTF']
                        break
            
            for b in atn_bids:
                bid_hour_start_ms = int((b.message_persisted_ms + 3599_999) // 3_600_000 * 3_600) * 1000
                if bid_hour_start_ms == hour_start_ms:
                    atn_bid = AtnBid(**b.payload)
                    bid = str(extract_pq_pairs(atn_bid))
                    break

            row = HourlyElectricity(
                g_node_alias=reports[0].from_alias,
                short_alias=self.house_alias,
                hour_start_s=int(hour_start_ms/1000),
                hp_kwh_el=hp_elec_in,
                hp_kwh_th=hp_heat_out,
                dist_kwh=dist_kwh,
                store_change_kwh=store_change_kwh,
                hp_avg_lwt=hp_avg_lwt,
                hp_avg_ewt=hp_avg_ewt,
                dist_avg_swt=dist_avg_swt,
                dist_avg_rwt=dist_avg_rwt,
                buffer_depth1_start=buffer_temps['buffer-depth1'],
                buffer_depth2_start=buffer_temps['buffer-depth2'],
                buffer_depth3_start=buffer_temps['buffer-depth3'],
                buffer_depth4_start=buffer_temps['buffer-depth4'],
                tank1_depth1_start=storage_temps['tank1-depth1'],
                tank1_depth2_start=storage_temps['tank1-depth2'],
                tank1_depth3_start=storage_temps['tank1-depth3'],
                tank1_depth4_start=storage_temps['tank1-depth4'],
                tank2_depth1_start=storage_temps['tank2-depth1'],
                tank2_depth2_start=storage_temps['tank2-depth2'],
                tank2_depth3_start=storage_temps['tank2-depth3'],
                tank2_depth4_start=storage_temps['tank2-depth4'],
                tank3_depth1_start=storage_temps['tank3-depth1'],
                tank3_depth2_start=storage_temps['tank3-depth2'],
                tank3_depth3_start=storage_temps['tank3-depth3'],
                tank3_depth4_start=storage_temps['tank3-depth4'],
                relay_3_pulled_fraction=relay3_pulled_fraction,
                relay_5_pulled_fraction=relay5_pulled_fraction,
                relay_6_pulled_fraction=relay6_pulled_fraction,
                relay_9_pulled_fraction=relay9_pulled_fraction,
                zone1_heatcall_fraction=zone1_heatcall_fraction,
                zone2_heatcall_fraction=zone2_heatcall_fraction,
                zone3_heatcall_fraction=zone3_heatcall_fraction,
                zone4_heatcall_fraction=zone4_heatcall_fraction,
                oat_f=oat_f,
                ws_mph=ws_mph,
                total_usd_per_mwh=total_usd_per_mwh,
                flo=flo_tf,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                intermediate_power_kw=intermediate_power_kw,
                intermediate_rswt=intermediate_rswt,
                dd_power_kw=dd_power_kw,
                dd_rswt=dd_rswt,
                dd_delta_t=dd_delta_t,
                bid=bid,
            )
            batch_rows.append(row)
        
        try:
            self.session_gbo.add_all(batch_rows)
            self.session_gbo.commit()
            print(f"Successfully inserted {len(batch_rows)} new batch_rows")

        except Exception as e:
            if 'hour_house_unique' in str(e) or "hourly_electricity_pkey" in str(e):
                print("Some batch_rows already exist in the database, filtering them out...")
                self.session_gbo.rollback()
                conflicting_rows = []
                for row in batch_rows:
                    try:
                        self.session_gbo.add(row)
                        self.session_gbo.commit()
                    except Exception:
                        self.session_gbo.rollback()
                        conflicting_rows.append(row)
                non_conflicting_batch_rows = [row for row in batch_rows if row not in conflicting_rows]
                if non_conflicting_batch_rows:
                    self.session_gbo.add_all(non_conflicting_batch_rows)
                    self.session_gbo.commit()
                    print(f"Successfully inserted {len(non_conflicting_batch_rows)} new batch_rows")
                else:
                    print("All batch_rows already existed in the database")
            else:
                self.session_gbo.rollback()
                raise Exception(f"Unexpected error: {e}")

    def unix_ms_to_date(self, time_ms):
        return str(pendulum.from_timestamp(time_ms/1000, tz=self.timezone_str).format('YYYY-MM-DD HH:mm'))
    
    def to_fahrenheit(self, t):
        return round(t*9/5+32,1)
    

if __name__ == '__main__':
    houses_to_generate = ['oak', 'fir', 'maple', 'elm', 'beech']
    timezone = 'America/New_York'

    for house in houses_to_generate:        
        start_year = pendulum.now(timezone).subtract(days=1).year
        start_month = pendulum.now(timezone).subtract(days=1).month
        start_day = pendulum.now(timezone).subtract(days=1).day
        end_year = pendulum.now(timezone).year
        end_month = pendulum.now(timezone).month
        end_day = pendulum.now(timezone).day
        
        start_ms = pendulum.datetime(start_year, start_month, start_day, tz=timezone).timestamp()*1000
        end_ms = pendulum.datetime(end_year, end_month, end_day, tz=timezone).timestamp()*1000
        
        s = HourlyData(house, start_ms, end_ms, timezone)
        s.generate_hourly_data()
        print(f"Done.")

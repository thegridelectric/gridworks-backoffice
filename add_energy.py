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
import matplotlib.pyplot as plt

dotenv.load_dotenv()

engine_gbo = create_engine(os.getenv("GBO_DB_URL"))
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
    
    __table_args__ = (
        UniqueConstraint('hour_start_s', 'g_node_alias', name='hour_house_unique'),
    )

# Drop existing tables
Base.metadata.drop_all(engine_gbo)
Base.metadata.create_all(engine_gbo)
if os.path.exists(f"energy_data_beech.csv"):
    os.remove(f"energy_data_beech.csv")

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
        whitewire_threshold_watts = {'beech': 100, 'elm': 0.9, 'default': 20}
        if self.house_alias in whitewire_threshold_watts:
            self.whitewire_threshold = whitewire_threshold_watts[self.house_alias]
        else:
            self.whitewire_threshold = whitewire_threshold_watts['default']
        primary_pump_gpm = {'beech': 5.5, 'default': 5}
        if self.house_alias in primary_pump_gpm:
            self.primary_pump_gpm = primary_pump_gpm[self.house_alias]
        else:
            self.primary_pump_gpm = primary_pump_gpm['default']
        store_pump_gpm = {'beech': 1.5, 'default': 1.5}
        if self.house_alias in store_pump_gpm:
            self.store_pump_gpm = store_pump_gpm[self.house_alias]
        else:
            self.store_pump_gpm = store_pump_gpm['default']
        self.data_format = {
            'g_node_alias': [],
            'short_alias': [],
            'hour_start_ms': [],
            'hp_kwh_el': [],
            'hp_kwh_th': [],
            'dist_kwh': [],
            'store_change_kwh': [],
            'hp_avg_lwt': [],
            'hp_avg_ewt': [],
            'dist_avg_swt': [],
            'dist_avg_rwt': [],
            'buffer_depth1_start': [],
            'buffer_depth2_start': [],
            'buffer_depth3_start': [],
            'buffer_depth4_start': [],
            'tank1_depth1_start': [],
            'tank1_depth2_start': [],
            'tank1_depth3_start': [],
            'tank1_depth4_start': [],
            'tank2_depth1_start': [],
            'tank2_depth2_start': [],
            'tank2_depth3_start': [],
            'tank2_depth4_start': [],
            'tank3_depth1_start': [],
            'tank3_depth2_start': [],
            'tank3_depth3_start': [],
            'tank3_depth4_start': [],
            'relay_3_pulled_fraction': [],
            'relay_5_pulled_fraction': [],
            'relay_6_pulled_fraction': [],
            'relay_9_pulled_fraction': [],
            'zone1_heatcall_fraction': [],
            'zone2_heatcall_fraction': [],
            'zone3_heatcall_fraction': [],
            'zone4_heatcall_fraction': [],
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
        BATCH_SIZE = 20
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

            # Sort data by channels
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

            # Get synchronous data for required data channels
            required_channels = [
                'hp-idu-pwr', 'hp-odu-pwr', 'hp-lwt', 'hp-ewt', 'primary-flow', 'primary-pump-pwr',
                'store-flow', 'store-hot-pipe', 'store-cold-pipe', 'store-pump-pwr',
                'charge-discharge-relay3', 'hp-failsafe-relay5', 'hp-scada-ops-relay6', 'store-pump-failsafe-relay9',
                'dist-swt', 'dist-rwt', 'dist-flow'
            ] + [
                x for x in channels if 'zone' in x and 'whitewire' in x
            ]
            hp_critical_channels = ['hp-idu-pwr', 'hp-odu-pwr']
            hp_required_channels = [x for x in required_channels if 'hp' in x or 'primary-flow' in x]
            store_required_channels = [x for x in required_channels if 'flow' in x or 'store' in x or 'relay3' in x and 'pwr' not in x]
            dist_required_channels = [x for x in required_channels if 'dist' in x]

            timestep_seconds = 1
            num_points = int((hour_end_ms - hour_start_ms) / (timestep_seconds * 1000) + 1)

            csv_times = np.linspace(hour_start_ms, hour_end_ms, num_points)
            csv_times_dt = pd.to_datetime(csv_times, unit='ms', utc=True)
            csv_times_dt = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in csv_times_dt]

            csv_values = {'times': csv_times}
            for channel in required_channels:
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

            # Calculations from synchronous data
            df = pd.DataFrame(csv_values)
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

            # Heat pump energy
            if not [c for c in hp_critical_channels if c not in csv_values]:
                df['hp_power'] = df['hp-idu-pwr'] + df['hp-odu-pwr']
                hp_elec_in = round(float(np.mean(df['hp_power'])/1000),2)
            else:
                print(f"Missing critical channels: {hp_critical_channels}")
                continue
            if (not [c for c in hp_required_channels if c not in csv_values]
                or ('primary-pump-pwr' in csv_values and not [c for c in hp_required_channels if c not in csv_values and 'primary-flow' not in c])):

                if 'primary-pump-pwr' in csv_values and hour_end_ms < pendulum.datetime(2025,1,1,tz=self.timezone_str).timestamp()*1000:
                    primary_flow_processed = []
                    last_correct_gpm = self.primary_pump_gpm
                    # Missing primary flow
                    if 'primary-flow' not in csv_values:
                        for i in range(len(df)):
                            value_watts = float(df['primary-pump-pwr'][i])
                            pump_on = value_watts > 10
                            if pump_on:
                                value_gpm = last_correct_gpm
                            else:
                                value_gpm = 0
                            primary_flow_processed.append(value_gpm*100)
                    
                    # Process primary flow
                    else:
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
                
                    df['primary-flow-processed'] = primary_flow_processed
                    
                    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    # if 'primary-flow' in csv_values:
                    #     ax.plot(df['primary-flow'], label='primary-flow', color='purple', alpha=0.5, linestyle='--')
                    # ax.plot(df['primary-flow-processed'], label='primary-flow-processed', color='red', alpha=0.5)
                    # ax2 = ax.twinx()    
                    # ax2.plot(df['primary-pump-pwr'], label='primary-pump-pwr', color='pink', alpha=0.6, linestyle='--')
                    # ax.set_ylim(-50, 1000)
                    # ax2.set_ylim(-5, 100)
                    # ax.legend()
                    # ax2.legend()
                    # plt.title(f"{self.unix_ms_to_date(hour_start_ms)}")
                    # plt.show()

                    df['primary-flow'] = df['primary-flow-processed']

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
                else:
                    print("There are no non-NaN cumulative heat values")
                    hp_heat_out = None
                hp_avg_lwt = self.to_fahrenheit(float(np.mean(df['hp-lwt'])/1000))
                hp_avg_ewt = self.to_fahrenheit(float(np.mean(df['hp-ewt'])/1000))
            else:
                print(f"Missing HP required channels: {[c for c in hp_required_channels if c not in csv_values]}")

            # Distribution energy
            if not [c for c in dist_required_channels if c not in csv_values]:
                df['dist_flow_kgs'] = df['dist-flow'] / 100 / 60 * 3.78541
                df['dist_lift_C'] = df['dist-swt'] - df['dist-rwt']
                df['dist_lift_C'] = df['dist_lift_C']/1000
                df['dist_heat_power_kW'] = [m*4187*lift/1000 for lift, m in zip(df['dist_lift_C'], df['dist_flow_kgs'])]
                df['dist_cumulative_heat_kWh'] = df['dist_heat_power_kW'].cumsum()
                df['dist_cumulative_heat_kWh'] = df['dist_cumulative_heat_kWh'] / 3600 * timestep_seconds
                dist_kwh = round(list(df['dist_cumulative_heat_kWh'])[-1] - list(df['dist_cumulative_heat_kWh'])[0],2)   
                dist_avg_swt = self.to_fahrenheit(float(np.mean(df['dist-swt'])/1000))
                dist_avg_rwt = self.to_fahrenheit(float(np.mean(df['dist-rwt'])/1000))

            # Storage energy
            if (not [c for c in store_required_channels if c not in csv_values]
                or ('store-pump-pwr' in csv_values and not [c for c in store_required_channels if c not in csv_values and 'store-flow' not in c and 'primary-flow' not in c])):

                if 'store-pump-pwr' in csv_values and hour_end_ms < pendulum.datetime(2025,1,1,tz=self.timezone_str).timestamp()*1000:
                    store_flow_processed = []
                    last_correct_gpm = self.store_pump_gpm
                    # Missing store flow
                    if 'store-flow' not in csv_values:
                        for i in range(len(df)):
                            value_watts = float(df['store-pump-pwr'][i])
                            pump_on = value_watts > 5
                            if pump_on:
                                value_gpm = last_correct_gpm
                            else:
                                value_gpm = 0
                            store_flow_processed.append(value_gpm*100)
                    
                    # Process store flow
                    else:
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
                
                    df['store-flow-processed'] = store_flow_processed

                    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    # if 'store-flow' in csv_values:
                    #     ax.plot(df['store-flow'], label='store-flow', color='purple', alpha=0.5, linestyle='--')
                    # ax.plot(df['store-flow-processed'], label='store-flow-processed', color='red', alpha=0.5)
                    # ax2 = ax.twinx()    
                    # ax2.plot(df['store-pump-pwr'], label='store-pump-pwr', color='pink', alpha=0.6, linestyle='--')
                    # ax.set_ylim(-50, 1000)
                    # ax2.set_ylim(-5, 100)
                    # ax.legend()
                    # ax2.legend()
                    # plt.title(f"{self.unix_ms_to_date(hour_start_ms)}")
                    # plt.show()

                    df['store-flow'] = df['store-flow-processed']

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
            else:
                print(f"Missing store required channels: {[c for c in store_required_channels if c not in csv_values]}")
                print(f"Missing store pump power: {'store-pump-pwr' not in csv_values}")

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

            # Relays
            if [c for c in csv_values if 'relay3' in c]:
                df['relay3_cumulative'] = df['charge-discharge-relay3'].cumsum()
                relay3_pulled_fraction = round(list(df['relay3_cumulative'])[-1] / len(df['relay3_cumulative']), 2)
            if [c for c in csv_values if 'relay5' in c]:
                df['relay5_cumulative'] = df['hp-failsafe-relay5'].cumsum()
                relay5_pulled_fraction = round(list(df['relay5_cumulative'])[-1] / len(df['relay5_cumulative']), 2)
            if [c for c in csv_values if 'relay6' in c]:
                df['relay6_cumulative'] = df['hp-scada-ops-relay6'].cumsum()
                relay6_pulled_fraction = round(list(df['relay6_cumulative'])[-1] / len(df['relay6_cumulative']), 2)
            if [c for c in csv_values if 'relay9' in c]:
                df['relay9_cumulative'] = df['store-pump-failsafe-relay9'].cumsum()
                relay9_pulled_fraction = round(list(df['relay9_cumulative'])[-1] / len(df['relay9_cumulative']), 2)

            # Zones
            whitewire_channels = [c for c in csv_values if 'whitewire' in c]
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

            # Cumulative energy balance
            # if relay3_pulled_fraction > 0.96:
            #     plt.plot(df['hp-lwt'].apply(lambda x: self.to_fahrenheit(x/1000)), label='HP LWT', color='tab:red')
            #     plt.plot(df['store-hot-pipe'].apply(lambda x: self.to_fahrenheit(x/1000)), label='Store hot pipe', color='tab:red', linestyle='--')
            #     plt.plot(df['hp-ewt'].apply(lambda x: self.to_fahrenheit(x/1000)), label='HP EWT', color='tab:blue')
            #     plt.plot(df['store-cold-pipe'].apply(lambda x: self.to_fahrenheit(x/1000)), label='Store cold pipe', color='tab:blue', linestyle='--')
            #     plt.legend()
            #     plt.show()
            #     plt.plot(df['lift_C']*9/5, label='HP lift', color='tab:blue')
            #     plt.plot(-df['store_lift_C']*9/5, label='Store lift', color='tab:orange')
            #     plt.plot(df['pipe_lift_C']*9/5, label='Pipe lift', color='tab:green')
            #     plt.legend()
            #     plt.show()
            #     plt.plot(df['cumulative_heat_kWh'], label='Heat Pump', color='tab:blue', alpha=0.7)
            #     plt.plot(-df['store_cumulative_heat_kWh'], label='Storage', color='tab:orange', alpha=0.7)
            #     plt.plot(df['pipe_cumulative_heat_kWh'], label='Pipe (HP->Storage)', color='tab:green', alpha=0.7)
            #     df['total'] = -df['store_cumulative_heat_kWh'] + df['pipe_cumulative_heat_kWh']
            #     plt.plot(df['total'], label='Storage + Pipe', color='tab:red', alpha=0.4, linestyle='dashed', linewidth=5)
            #     plt.xlabel("Time since hour start (seconds)")
            #     plt.ylabel("Cumulative heat [kWh]")
            #     plt.legend()
            #     plt.show()

            print(f"{self.unix_ms_to_date(hour_start_ms)} - HP: {hp_elec_in} kWh_e, {hp_heat_out} kWh_th")

            row = [
                reports[0].from_alias, 
                self.house_alias, 
                hour_start_ms, 
                hp_elec_in, 
                hp_heat_out,
                dist_kwh,
                store_change_kwh, 
                hp_avg_lwt,
                hp_avg_ewt,
                dist_avg_swt,
                dist_avg_rwt,
                buffer_temps['buffer-depth1'],
                buffer_temps['buffer-depth2'],
                buffer_temps['buffer-depth3'],
                buffer_temps['buffer-depth4'],
                storage_temps['tank1-depth1'],
                storage_temps['tank1-depth2'],
                storage_temps['tank1-depth3'],
                storage_temps['tank1-depth4'],
                storage_temps['tank2-depth1'],
                storage_temps['tank2-depth2'],
                storage_temps['tank2-depth3'],
                storage_temps['tank2-depth4'],
                storage_temps['tank3-depth1'],
                storage_temps['tank3-depth2'],
                storage_temps['tank3-depth3'],
                storage_temps['tank3-depth4'],
                relay3_pulled_fraction,
                relay5_pulled_fraction,
                relay6_pulled_fraction,
                relay9_pulled_fraction,
                zone1_heatcall_fraction,
                zone2_heatcall_fraction,
                zone3_heatcall_fraction,
                zone4_heatcall_fraction,
            ]
            row = [x if x is not None else np.nan for x in row]
            formatted_data.loc[len(formatted_data)] = row 

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
    houses_to_generate = ['beech']#, 'oak', 'fir', 'maple', 'elm']
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

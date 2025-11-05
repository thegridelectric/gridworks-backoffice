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


class EnergyDataset():
    def __init__(self, house_alias, start_ms, end_ms, timezone):
        engine = create_engine(os.getenv("GJK_DB_URL"))
        Session = sessionmaker(bind=engine)
        self.session = Session()
        SessionGbo = sessionmaker(bind=engine_gbo)
        self.session_gbo = SessionGbo()
        self.house_alias = house_alias
        self.dataset_file = f"pico_reliability_data_{self.house_alias}.csv"
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.timezone_str = timezone
        self.stop_flow_processing_date_ms = pendulum.datetime(2025,1,1,tz=self.timezone_str).timestamp()*1000
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
            'buffer_depth1': [],
            'buffer_depth2': [],
            'buffer_depth3': [],
            'buffer_depth4': [],
            'tank1_depth1': [],
            'tank1_depth2': [],
            'tank1_depth3': [],
            'tank1_depth4': [],
            'tank2_depth1': [],
            'tank2_depth2': [],
            'tank2_depth3': [],
            'tank2_depth4': [],
            'tank3_depth1': [],
            'tank3_depth2': [],
            'tank3_depth3': [],
            'tank3_depth4': [],
            'relay1_activations': [],
            'relay1_all_activations': [],
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
        print(f"Data for {self.house_alias} starts at {self.unix_ms_to_date(self.start_ms)} ({self.start_ms})")

    def generate_dataset(self):
        print("\nGenerating dataset...")
        self.find_first_date()
        existing_dataset_dates = []
        # if os.path.exists(self.dataset_file):
        #     print(f"Found existing dataset: {self.dataset_file}")
        #     df = pd.read_csv(self.dataset_file)
        #     existing_dataset_dates = [int(x) for x in list(df['hour_start_ms'])]

        # Add data in batches of BATCH_SIZE hours
        SMALL_BATCH_SIZE = 20
        LARGE_BATCH_SIZE = 100
        using_small_batches = True

        batch_start_ms = int(pendulum.from_timestamp(self.start_ms/1000, tz=self.timezone_str).replace(hour=0, minute=0, microsecond=0).timestamp()*1000)
        batch_end_ms = batch_start_ms + SMALL_BATCH_SIZE*3600*1000
        now_ms = int(time.time()*1000)
        
        while batch_start_ms < min(self.end_ms, now_ms):
            batch_end_ms = min(batch_end_ms, now_ms, self.end_ms)
            if existing_dataset_dates and int(batch_end_ms-3600*1000) <= max(existing_dataset_dates):
                print("Batch is already in data")
                batch_start_ms += SMALL_BATCH_SIZE*3600*1000
                batch_end_ms += SMALL_BATCH_SIZE*3600*1000
            else:
                # We are processing all the data: small batches
                if batch_start_ms < self.stop_flow_processing_date_ms:
                    self.add_data(batch_start_ms, batch_end_ms)
                    batch_start_ms += SMALL_BATCH_SIZE*3600*1000
                    batch_end_ms += SMALL_BATCH_SIZE*3600*1000
                
                # All the necessary data has been processed: move to large batches
                elif batch_start_ms >= self.stop_flow_processing_date_ms:
                    self.add_data(batch_start_ms, batch_end_ms)
                    if using_small_batches:
                        batch_start_ms += SMALL_BATCH_SIZE*3600*1000
                        batch_end_ms += LARGE_BATCH_SIZE*3600*1000
                        using_small_batches = False
                    else:
                        batch_start_ms += LARGE_BATCH_SIZE*3600*1000
                        batch_end_ms += LARGE_BATCH_SIZE*3600*1000

    def add_data(self, batch_start_ms, batch_end_ms):
        st = time.time()
        print(f"\nGathering reports from: {self.unix_ms_to_date(batch_start_ms)} to {self.unix_ms_to_date(batch_end_ms)}...")
        
        reports: List[MessageSql] = self.session.query(MessageSql).filter(
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            or_(
                MessageSql.message_type_name == "batched.readings",
                MessageSql.message_type_name == "report",
                MessageSql.message_type_name == "layout.lite"
            ),
            MessageSql.message_persisted_ms >= batch_start_ms - 7*60*1000,
            MessageSql.message_persisted_ms <= batch_end_ms + 7*60*1000,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()

        reports = [r for r in reports if r.message_type_name != 'layout.lite']
        layout_lite_reports = [r for r in reports if r.message_type_name == 'layout.lite']
        layout_lite_received_times = [r.message_persisted_ms for r in layout_lite_reports]
        
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
            # print(f"Layout.lite received times: {layout_lite_received_times}")

            # Figure out which readings hav
            tank_channels = [
                'buffer-depth1', 'buffer-depth2', 'buffer-depth3', 'buffer-depth4'] + [
                'tank1-depth1', 'tank1-depth2', 'tank1-depth3', 'tank1-depth4', 
                'tank2-depth1', 'tank2-depth2', 'tank2-depth3', 'tank2-depth4', 
                'tank3-depth1', 'tank3-depth2', 'tank3-depth3', 'tank3-depth4'
            ]
            tank_temps = {x: False for x in tank_channels}
            for channel in tank_temps:
                if channel not in channels or len(channels[channel]['times']) < 2:
                    continue
                all_times = [hour_start_ms] + channels[channel]['times'] + [hour_end_ms]
                max_time_gap = max(t2-t1 for t1, t2 in zip(all_times[:-1], all_times[1:]))
                if max_time_gap/1e3 < 15*60:
                    tank_temps[channel] = True
                        
            # Count the number of relay 1 activations more than 5 minutes after any layout.lite message
            relay1_activations = 0
            relay1_all_activations = 0
            if 'vdc-relay1' in channels:
                relay1_times = channels['vdc-relay1']['times']
                relay1_values = channels['vdc-relay1']['values']
                for i in range(len(relay1_times)):
                    if relay1_values[i] > 0:
                        relay1_all_activations += 1
                        # There are no layout.lite messages: we did not recently reboot
                        if not layout_lite_received_times:
                            relay1_activations += 1
                            continue
                        print("WOOOOOOOOOOO")
                        min_distance = 1e20
                        for j in range(len(layout_lite_received_times)):
                            distance = abs(relay1_times[i] - layout_lite_received_times[j])
                            min_distance = min(min_distance, distance)
                        print(f"Min time between relay1 activation and layout.lite message: {min_distance/1000} seconds")
                        # We didn't recently reboot, count the activation
                        if min_distance > 5*60*1000:
                            relay1_activations += 1

            row = [
                reports[0].from_alias, 
                self.house_alias, 
                hour_start_ms, 
                tank_temps['buffer-depth1'],
                tank_temps['buffer-depth2'],
                tank_temps['buffer-depth3'],
                tank_temps['buffer-depth4'],
                tank_temps['tank1-depth1'],
                tank_temps['tank1-depth2'],
                tank_temps['tank1-depth3'],
                tank_temps['tank1-depth4'],
                tank_temps['tank2-depth1'],
                tank_temps['tank2-depth2'],
                tank_temps['tank2-depth3'],
                tank_temps['tank2-depth4'],
                tank_temps['tank3-depth1'],
                tank_temps['tank3-depth2'],
                tank_temps['tank3-depth3'],
                tank_temps['tank3-depth4'],
                relay1_activations,
                relay1_all_activations
            ]

            row = [x if x is not None else np.nan for x in row]
            formatted_data.loc[len(formatted_data)] = row 

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
    houses_to_generate = ['maple', 'elm']
    for house in houses_to_generate:
        generate(
            house_alias=house, 
            start_year=2025, 
            start_month=3, 
            start_day=1,
            end_year=2025,
            end_month=5,
            end_day=31
        )

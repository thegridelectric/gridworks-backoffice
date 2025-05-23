import pendulum
import dotenv
import json
from gjk.models import MessageSql
from gjk.config import Settings
from sqlalchemy import create_engine, asc
from sqlalchemy.orm import sessionmaker
from gjk.named_types import TicklistHallReport, TicklistReedReport, ChannelReadings, TicklistHall, TicklistReed
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path

settings = Settings(_env_file=dotenv.find_dotenv())
engine = create_engine(settings.db_url.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()

maple_primary_pico = "pico_481731"
maple_dist_pico = "pico_497d35"
oak_dist_pico = "pico_974532"
selected_pico = maple_dist_pico
hall_meter = True

output_file = Path("ticklists_data.json")

if output_file.exists():
    print(f"Loading existing ticklists from {output_file}")
    with open(output_file, "r") as f:
        ticklists_data = json.load(f)
    ticklists = [TicklistHallReport(**t) for t in ticklists_data]
    print(f"Loaded {len(ticklists)} ticklists")

else:
    start_ms = pendulum.datetime(2025, 5, 23, 10, 30,  tz='America/New_York').timestamp()*1000
    # end_ms = pendulum.datetime(2025, 5, 23, 8, 0, 0, tz='America/New_York').timestamp()*1000
    print("Finding data...")
    if hall_meter:
        print("Finding hall meter ticklists")
        messages = session.query(MessageSql).filter(
            MessageSql.message_type_name == "ticklist.hall.report",
            MessageSql.message_persisted_ms >= start_ms,
            # MessageSql.message_persisted_ms <= end_ms,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()
        ticklists = [TicklistHallReport(**m.payload) for m in messages]
        print(f"Found {len(ticklists)} ticklists. \n\nFirst one:\n{ticklists[0]}")
    else:
        messages = session.query(MessageSql).filter(
            MessageSql.message_type_name == "ticklist.reed.report",
            MessageSql.message_persisted_ms >= start_ms,
            # MessageSql.message_persisted_ms <= end_ms,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()
        ticklists = [TicklistReedReport(**m.payload) for m in messages]
        print(f"Found {len(ticklists)} ticklists. \n\nFirst one:\n{ticklists[0]}")

    ticklists_data = [t.model_dump() for t in ticklists]
    with open(output_file, "w") as f:
        json.dump(ticklists_data, f, indent=2)
    print(f"Saved {len(ticklists)} ticklists to {output_file}")

# Process the ticklists one by one like in the real world
ticklists_to_process = [t for t in ticklists if t.ticklist.hw_uid == selected_pico]
print(f"ready to process {len(ticklists_to_process)} ticklists")

class ApiFlowModule():
    def __init__(self):
        self.slow_turner = False
        self.nano_timestamps: List[int] = []
        self.latest_tick_ns = None
        self.latest_hz = None
        self.latest_gpm = None

        self.AsyncCaptureThresholdGpmTimes100 = 20
        self.ConstantGallonsPerTick = 0.0009
        self.NoFlowMs = 250
        self.hz_calculation_method = "BasicExpWeightedAvg"
        self.exp_alpha = 0.2
        self.capture_s = 20

        self.gpm_readings = {'times': [], 'values': []}
        self.hz_readings = {'times': [], 'values': []}
        self.main()

    def add_readings(self, channel, times, values):
        times = [pendulum.from_timestamp(x/1000, tz='America/New_York') for x in times]
        if channel=='gpm':
            self.gpm_readings['times'].extend(times)
            self.gpm_readings['values'].extend(values)
        elif channel=='hz':
            self.hz_readings['times'].extend(times)
            self.hz_readings['values'].extend(values)
    
    def publish_zero_flow(self):
        print("publish_zero_flow")
        self.latest_gpm = 0
        self.latest_hz = 0
        # Set the timestamp for the zero reading just after (100ms) the latest tick received
        if self.latest_tick_ns:
            zero_flow_ms = int((self.latest_tick_ns+1e8) / 1e6)
        else:
            return
        self.add_readings('gpm', [zero_flow_ms], [0])
        self.add_readings('hz', [zero_flow_ms], [0])

    def update_timestamps_for_hall(self, data: TicklistHall, scada_received_unix_ms: int) -> None:
        pi_time_received_post = scada_received_unix_ms*1e6
        pico_time_before_post = data.pico_before_post_timestamp_nano_second
        pico_time_delay_ns = pi_time_received_post - pico_time_before_post
        self.nano_timestamps = sorted(
            list(
                set(
                    [
                        data.first_tick_timestamp_nano_second + x*1e3 + pico_time_delay_ns
                        for x in data.relative_microsecond_list
                    ]
                )
            )
        )

    def publish_first_frequency(self):
        if not self.latest_tick_ns:
            return
        # seconds_since_last_tick = (self.nano_timestamps[0]-self.latest_tick_ns)*1e9
        # if not self.slow_turner and seconds_since_last_tick > 5:
        #     self.publish_zero_flow()
        #     return
        # Frequency between last timestamp from prev list and first from current list
        first_frequency = 1/(self.nano_timestamps[0]-self.latest_tick_ns)*1e9
        # Not a slow turner: if there was less than 5s since last tick ignore this
        if not self.slow_turner and first_frequency > 1/5:
            return
        # Compute GPM and publish
        gallons_per_tick = self.ConstantGallonsPerTick
        self.latest_gpm = first_frequency * 60 * gallons_per_tick        
        self.latest_hz = first_frequency
        self.add_readings('gpm', [self.latest_tick_ns/1e6], [self.latest_gpm])
        self.add_readings('hz', [self.latest_tick_ns/1e6], [self.latest_hz])

    def get_micro_hz_readings(self, filtering: bool) -> List[float]:
        if len(self.nano_timestamps)==0:
            raise ValueError("Should not call get_hz_readings with an empty ticklist!")

        # Single tick
        if len(self.nano_timestamps)==1:
            if self.latest_tick_ns is not None:
                frequency_hz = 1e9 / (self.nano_timestamps[0] - self.latest_tick_ns)
            else:
                frequency_hz = 0
            if self.slow_turner:
                micro_hz_readings = ChannelReadings(
                    ChannelName='gpm',
                    ValueList=[int(frequency_hz * 1e6)],
                    ScadaReadTimeUnixMsList=[int(self.nano_timestamps[0]/1e6)]
                )
            else:
                micro_hz_readings = ChannelReadings(
                    ChannelName='gpm',
                    ValueList=[int(frequency_hz * 1e6), 0],
                    ScadaReadTimeUnixMsList=[int(self.nano_timestamps[0]/1e6), int(self.nano_timestamps[0]/1e6)+100]
                )
            self.latest_tick_ns = self.nano_timestamps[-1]
            self.latest_hz = frequency_hz if self.slow_turner else 0
            return micro_hz_readings
        
        # Post flow between the latest tick and the first tick
        # if self.latest_tick_ns:
        #     self.publish_first_frequency()

        # Sort timestamps and compute frequencies
        timestamps = sorted(self.nano_timestamps)
        frequencies = [1/(t2-t1)*1e9 for t1,t2 in zip(timestamps[:-1], timestamps[1:])]
        timestamps = timestamps[:-1]

        if not self.slow_turner:
            # Remove outliers
            min_hz, max_hz = 0, 500
            tf_pairs = [(t,f) for t,f in zip(timestamps, frequencies) if f<=max_hz and f>=min_hz]
            timestamps = [x[0] for x in tf_pairs]
            frequencies = [x[1] for x in tf_pairs]
            if not timestamps:
                return ChannelReadings(
                    ChannelName='',
                    ValueList=[],
                    ScadaReadTimeUnixMsList=[],
                )

            # Add 0 flow when there is more than no_flow_ms between two points
            new_timestamps, new_frequencies = [], []
            for i in range(len(timestamps) - 1):
                new_timestamps.append(timestamps[i]) 
                new_frequencies.append(frequencies[i])  
                if timestamps[i+1] - timestamps[i] > self.NoFlowMs*1e6:
                    step_20ms = 0.02*1e9
                    while new_timestamps[-1] + step_20ms < timestamps[i+1]:
                        new_timestamps.append(new_timestamps[-1] + step_20ms)
                        new_frequencies.append(0.001)
            new_timestamps.append(timestamps[-1])
            new_frequencies.append(frequencies[-1])
            sorted_times_values = sorted(zip(new_timestamps, new_frequencies))
            timestamps, frequencies = zip(*sorted_times_values)

        # First reading
        first_reading = False
        if self.latest_hz is None:
            first_reading = True
            self.latest_hz = frequencies[0]

        # No processing for slow turners
        if self.slow_turner or not filtering:
            sampled_timestamps = timestamps
            smoothed_frequencies = frequencies
            self.latest_hz = smoothed_frequencies[-1]
            self.latest_tick_ns = sorted(self.nano_timestamps)[-1]
            return ChannelReadings(
                ChannelName='gpm',
                ValueList=[int(x*1e6) for x in smoothed_frequencies],
                ScadaReadTimeUnixMsList=[int(x/1e6) for x in sampled_timestamps],
            )

        # [Processing] Exponential weighted average
        elif self.hz_calculation_method == "BasicExpWeightedAvg":
            alpha = self.exp_alpha
            smoothed_frequencies = [self.latest_hz]*len(frequencies)
            for t in range(len(frequencies)-1):
                smoothed_frequencies[t+1] = (1-alpha)*smoothed_frequencies[t] + alpha*frequencies[t+1]
            sampled_timestamps = timestamps

        # [Processing] Butterworth filter
        # elif self.hz_calculation_method == "BasicButterWorth":
        #     if len(frequencies) > 20:
        #         # Add the last recorded frequency before the filtering (avoids overfitting the first point)
        #         timestamps = [timestamps[0]-0.01*1e9] + list(timestamps)
        #         frequencies = [self.latest_hz] + list(frequencies)
        #         # Re-sample time at sampling frequency f_s
        #         f_s = 5 * max(frequencies)
        #         sampled_timestamps = np.linspace(min(timestamps), max(timestamps), int((max(timestamps)-min(timestamps))/1e9 * f_s))
        #         # Re-sample frequency accordingly using a linear interpolaton
        #         sampled_frequencies = np.interp(sampled_timestamps, timestamps, frequencies)
        #         # Butterworth low-pass filter
        #         b, a = butter_lowpass(N=5, Wn=self._component.gt.CutoffFrequency, fs=f_s)
        #         smoothed_frequencies = filtering(b, a, sampled_frequencies)
        #         # Remove points resulting from adding the first recorded frequency
        #         smoothed_frequencies = [
        #             smoothed_frequencies[i] 
        #             for i in range(len(smoothed_frequencies)) 
        #             if sampled_timestamps[i]>=timestamps[1]
        #         ]
        #         sampled_timestamps = [x for x in sampled_timestamps if x>=timestamps[1]]
        #     else:
        #         self.log(f"Warning: ticklist was too short ({len(frequencies)} instead of minimum 20) for butterworth.")
        #         sampled_timestamps = timestamps
        #         smoothed_frequencies = frequencies

        # Sanity checks after processing
        if not sampled_timestamps or len(sampled_timestamps) != len(smoothed_frequencies) :
            if not sampled_timestamps:
                glitch_summary = "Filtering resulted in a list of length 0"
            else:
                glitch_summary = "Sampled Timestamps and Smoothed Frequencies not the same length!"
            print(glitch_summary)
            if not sampled_timestamps:
                return ChannelReadings(
                    ChannelName='gpm',
                    ValueList=[],
                    ScadaReadTimeUnixMsList=[],
                )
            else:
                raise Exception("Sampled Timestamps and Smoothed Frequencies not the same length!")          
        
        # Record Hz on change
        threshold_gpm = self.AsyncCaptureThresholdGpmTimes100 / 100
        gallons_per_tick = self.ConstantGallonsPerTick
        threshold_hz = threshold_gpm / 60 / gallons_per_tick
        if first_reading:
            self.latest_hz = smoothed_frequencies[0]
        micro_hz_list = [int(self.latest_hz * 1e6)]
        unix_ms_times = [int(sampled_timestamps[0] / 1e6)]
        for i in range(1, len(smoothed_frequencies)):
            if abs(smoothed_frequencies[i] - micro_hz_list[-1]/1e6) > threshold_hz:
                micro_hz_list.append(int(smoothed_frequencies[i] * 1e6))
                unix_ms_times.append(int(sampled_timestamps[i] / 1e6))
        self.latest_hz = micro_hz_list[-1]/1e6
        self.latest_tick_ns = sorted(self.nano_timestamps)[-1]
        micro_hz_list = [x if x>0 else 0 for x in micro_hz_list]
        
        return ChannelReadings(
            ChannelName='gpm',
            ValueList=micro_hz_list,
            ScadaReadTimeUnixMsList=unix_ms_times,
        )
    
    def get_gpm_readings(self, micro_hz_readings: ChannelReadings):
        if not micro_hz_readings.value_list:
            self.log("Empty micro hz list in get_gpm_readings")
            return
        gallons_per_tick = self.ConstantGallonsPerTick
        hz_list = [x / 1e6 for x in micro_hz_readings.value_list]
        gpms = [x * 60 * gallons_per_tick for x in hz_list]
        self.latest_gpm = gpms[-1]
        self.add_readings('hz', micro_hz_readings.scada_read_time_unix_ms_list, [x/1e6 for x in micro_hz_readings.value_list])
        self.add_readings('gpm', micro_hz_readings.scada_read_time_unix_ms_list, gpms)

    def _process_ticklist_hall(self, data: TicklistHall, scada_received_unix_ms: int, filtering: bool):

        # Process empty ticklist
        if len(data.relative_microsecond_list)==0:
            if self.latest_gpm is None:
                print("Empty ticklist, first reading")
                self.publish_zero_flow()
            elif self.latest_gpm > self.AsyncCaptureThresholdGpmTimes100/100:
                print("Empty ticklist, different from latest reading")
                self.publish_zero_flow()
            else:
                print("Empty ticklist, same as latest reading")
            return

        # Get absolute timestamps and corresponding frequency/GPM readings
        self.update_timestamps_for_hall(data, scada_received_unix_ms)
        micro_hz_readings = self.get_micro_hz_readings(filtering)
        self.get_gpm_readings(micro_hz_readings)

    def main(self):

        if not ticklists_to_process:
            return
        
        for ticklist in ticklists_to_process:
            self._process_ticklist_hall(ticklist.ticklist, ticklist.scada_received_unix_ms, filtering=True)

        backup_times = self.gpm_readings['times'].copy()
        backup_values = self.gpm_readings['values'].copy()

        self.gpm_readings['times'] = []
        self.gpm_readings['values'] = []

        for ticklist in ticklists_to_process:
            self._process_ticklist_hall(ticklist.ticklist, ticklist.scada_received_unix_ms, filtering=False)
        
        plt.figure(figsize=(15,4))
        plt.step(self.gpm_readings['times'], self.gpm_readings['values'], where='post', alpha=0.3, color='gray', label='raw')
        plt.step(backup_times, backup_values, where='post', alpha=0.5, color='red', label='processed')
        plt.xticks(rotation=45)
        plt.ylim(0,10)
        plt.legend()
        plt.show()

d = ApiFlowModule()
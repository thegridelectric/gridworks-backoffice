import pandas as pd
import matplotlib.pyplot as plt
import pendulum

df = pd.read_csv('energy_data_oak.csv')

buffer_start_cols = [col for col in df.columns if col.startswith('buffer') and col.endswith('start')]
df['average_buffer_temp_start'] = df[buffer_start_cols].apply(lambda x: x.mean(), axis=1)
storage_start_cols = [col for col in df.columns if col.startswith('tank') and col.endswith('start')]
df['average_store_temp_start'] = df[storage_start_cols].apply(lambda x: x.mean(), axis=1)
df = df.drop(columns=buffer_start_cols + storage_start_cols)

df['hour_start_dt'] = pd.to_datetime(df['hour_start'])
dates_temp = [pendulum.datetime(x.year, x.month, x.day, x.hour, x.minute, tz='America/New_York') for x in df['hour_start_dt']]
df['hour_start_s'] = [int(x.timestamp()) for x in dates_temp]

df_shifted = df.shift(-1)
one_hour_mask = (df_shifted['hour_start_dt'] - df['hour_start_dt']).dt.total_seconds() == 3600
df['average_buffer_temp_end'] = pd.Series(index=df.index)
df.loc[one_hour_mask, 'average_buffer_temp_end'] = df_shifted.loc[one_hour_mask, 'average_buffer_temp_start']
df.dropna(inplace=True)

df['buffer_change_kwh'] = 120*3.79*4.187/3600*(df['average_buffer_temp_end']-df['average_buffer_temp_start'])*5/9
df['implied_heat_load'] = df['hp_kwh_th'] - df['store_change_kwh'] - df['buffer_change_kwh']
df.dropna(inplace=True)

# Find all rows with the same storage and buffer start as the first row
target_store_temp = list(df['average_store_temp_start'])[0]
target_buffer_temp = list(df['average_buffer_temp_start'])[0]
filtered = df[
    (df['average_store_temp_start'] >= target_store_temp - 1) &
    (df['average_store_temp_start'] <= target_store_temp + 1) &
    (df['average_buffer_temp_start'] >= target_buffer_temp - 1) &
    (df['average_buffer_temp_start'] <= target_buffer_temp + 1)
]
# display(filtered)

# Take the dataframe between two of those rows
df_same_start_and_end_state = df[:1600-1]

# Add weather data
weather_and_prices_df = pd.read_csv('../data/simulation_data.csv')
final_df = pd.merge(df_same_start_and_end_state, weather_and_prices_df, on='hour_start_s', how='inner')
final_df = final_df[['hour_start', 'hp_kwh_th', 'implied_heat_load', 'dist_kwh', 'oat_f', 'ws_mph']]
# display(final_df)

# Calculate load based on weather
SCADA_ALPHA=9.3
SCADA_BETA=-0.17
SCADA_GAMMA=0.0015
final_df['weather_load'] = SCADA_ALPHA + SCADA_BETA * final_df['oat_f'] + SCADA_GAMMA * final_df['ws_mph']

# Prints
total_hp_heat_out = round(sum(final_df['hp_kwh_th']),3)
print(f"HP: {total_hp_heat_out}")
total_implied_heat_load = round(sum(final_df['implied_heat_load']),3)
print(f"Implied: {total_implied_heat_load}")
total_weather_load = round(sum(final_df['weather_load']),3)
print(f"Weather and parameters: {total_weather_load}")

# Scale weather based load
scaling_factor = total_weather_load/total_hp_heat_out
final_df['weather_load_scaled'] = final_df['weather_load'] / scaling_factor
total_weather_load_scaled = round(sum(final_df['weather_load_scaled']),3)
print(f"Weather and parameters, scaled: {total_weather_load_scaled}")
print(f"Alpha: {SCADA_ALPHA} -> {round(SCADA_ALPHA/scaling_factor,2)}")
print(f"Beta: {SCADA_BETA} -> {round(SCADA_BETA/scaling_factor,2)}")
print(f"Alpha: {SCADA_GAMMA} -> {round(SCADA_GAMMA/scaling_factor,5)}")

final_df_cropped = final_df[0:10000]
plt.figure(figsize=(12,4))
plt.step(range(len(final_df_cropped)), final_df_cropped['hp_kwh_th'], label='HP heat output', alpha=0.4)
plt.step(range(len(final_df_cropped)), final_df_cropped['implied_heat_load'], label='Implied heat load', alpha=0.6)
plt.step(range(len(final_df_cropped)), final_df_cropped['weather_load_scaled'], label='Load using weather and scaling', alpha=0.6)
# plt.step(range(len(final_df_cropped)), final_df_cropped['dist_kwh'], label='Distribution', alpha=0.6)
plt.legend()
plt.ylim([-1,30])
plt.show()
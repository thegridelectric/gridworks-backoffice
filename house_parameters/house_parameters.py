import pandas as pd
import matplotlib.pyplot as plt
import pendulum
import statsmodels.formula.api as smf
import numpy as np
import os

if os.path.exists('house_parameters/house_parameters_by_month.csv'):
    os.remove('house_parameters/house_parameters_by_month.csv')

for house_alias in ['beech', 'fir', 'oak', 'maple', 'elm']:
    for month in [9,10,11,12,1,2,3,4,5]:

        month_print = f"0{month}/2025" if month < 9 else (f"{month}/2024" if month>9 else f"0{month}/2024")
        print(f"\n{house_alias} - {month_print}:")

        # Outputs from backoffice electricity use CSV
        df = pd.read_csv(f'house_parameters/{house_alias}_electricity_use.csv')

        # Get time in unix seconds to merge with weather data
        df['hour_start_dt'] = pd.to_datetime(df['hour_start'])
        dates_temp = [pendulum.datetime(x.year, x.month, x.day, x.hour, x.minute, tz='America/New_York') for x in df['hour_start_dt']]
        df['hour_start_s'] = [int(x.timestamp()) for x in dates_temp]
        df['month'] = [x.month for x in dates_temp]
        df = df[df['month'] == month]
        if len(df) == 0:
            print("No data for this month")
            continue

        # Get average buffer and storage start temperature
        buffer_start_cols = [col for col in df.columns if col.startswith('buffer') and col.endswith('start')]
        df['average_buffer_temp_start'] = df[buffer_start_cols].apply(lambda x: x.mean(), axis=1)
        df['average_buffer_temp_start'] = df['average_buffer_temp_start'].round(1)
        storage_start_cols = [col for col in df.columns if col.startswith('tank') and col.endswith('start')]
        df['average_store_temp_start'] = df[storage_start_cols].apply(lambda x: x.mean(), axis=1)
        df['average_store_temp_start'] = df['average_store_temp_start'].round(1)
        df = df.drop(columns=buffer_start_cols + storage_start_cols)

        # Get average buffer and storage end temperature
        df_shifted = df.shift(-1)
        one_hour_mask = (df_shifted['hour_start_dt'] - df['hour_start_dt']).dt.total_seconds() == 3600
        df['average_buffer_temp_end'] = pd.Series(index=df.index)
        df.loc[one_hour_mask, 'average_buffer_temp_end'] = df_shifted.loc[one_hour_mask, 'average_buffer_temp_start']
        df['average_store_temp_end'] = pd.Series(index=df.index)
        df.loc[one_hour_mask, 'average_store_temp_end'] = df_shifted.loc[one_hour_mask, 'average_store_temp_start']

        # Get the implied heat load
        df['buffer_change_kwh'] = 120*3.79*4.187/3600*(df['average_buffer_temp_end']-df['average_buffer_temp_start'])*5/9
        df['implied_house_kwh'] = df['hp_kwh_th'] - df['store_change_kwh'] - df['buffer_change_kwh']

        '''
        If you need the calculation to be exact, you should make it between two times when 
        the storage and buffer temperatures are the same. This ensures that all the energy
        put out by the heat pump is the same as the energy used to heat the house.
        There is still some error because some water might be left in the distribution system,
        thus contributing to underestimating the energy received by the distribution system.
        '''
        # Find all rows with the same storage and buffer start as the first row
        # target_store_temp = list(df['average_store_temp_start'])[0]
        # target_buffer_temp = list(df['average_buffer_temp_start'])[0]
        # filtered = df[
        #     (df['average_store_temp_start'] >= target_store_temp - 1) &
        #     (df['average_store_temp_start'] <= target_store_temp + 1) &
        #     (df['average_buffer_temp_start'] >= target_buffer_temp - 1) &
        #     (df['average_buffer_temp_start'] <= target_buffer_temp + 1)
        # ]
        # same_states_idx = list(filtered.index)
        # Take the dataframe between two of those rows
        # df = df[:same_states_idx[2]-1]

        # Add weather data (simulation data is in this format: date, hour_start_s, usd_mwh, oat_f, ws_mph)
        weather_and_prices_df = pd.read_csv('house_parameters/simulation_data.csv')
        final_df = pd.merge(df, weather_and_prices_df, on='hour_start_s', how='inner')

        # Keep only some of the columns
        final_df = final_df[['hour_start_dt', 'hp_kwh_th', 'implied_house_kwh', 'dist_kwh', 'oat_f', 'ws_mph']]

        # If there is no non-nan data, skip this month
        test = final_df.copy()
        test = test.dropna()
        if len(test)==0:
            print("No non-nan data")
            print(final_df)
            continue

        # Drop all rows with nan data
        final_df.dropna(inplace=True)

        # Scale the distribution energy to match the total heat pump energy
        total_hp = round(sum(final_df['hp_kwh_th']),2)
        total_dist = round(sum(final_df['dist_kwh']),2)
        if total_dist == 0:
            print("No distribution data")
            continue
        ratio = round(total_hp/total_dist,5)
        print(f"Total HP {total_hp}, total dist {total_dist}, ratio {ratio}")
        final_df['dist_kwh_scaled'] = final_df['dist_kwh']*ratio

        final_df['oat_f_shifted'] = 60 - final_df['oat_f']

        # plt.figure(figsize=(13,4))
        # plt.plot(final_df['hour_start_dt'], final_df['dist_kwh'], label='Distribution', alpha=0.6)
        # plt.plot(final_df['hour_start_dt'], final_df['dist_kwh_scaled'], label='Distribution scaled', alpha=0.6)
        # plt.ylabel('Heat [kWh]')
        # plt.legend()
        # plt.show()

        # Create train/test split with 80% training data
        train_size = int(len(final_df) * 0.8)
        train_df = final_df.sample(n=train_size, random_state=42)
        test_df = final_df.drop(train_df.index)

        formula = 'dist_kwh_scaled ~ oat_f + I(oat_f_shifted * ws_mph)'
        model = smf.ols(formula=formula, data=train_df).fit()

        # Parameters
        alpha = model.params['Intercept']
        beta = model.params['oat_f']
        gamma = model.params['I(oat_f_shifted * ws_mph)']
        print(f"- alpha = {alpha:.3f}")
        print(f"- beta = {beta:.3f}")
        print(f"- gamma = {gamma:.5f}")

        # Standard errors
        alpha_se = model.bse['Intercept']
        beta_se = model.bse['oat_f']
        gamma_se = model.bse['I(oat_f_shifted * ws_mph)']
        print(f"- alpha SE = {alpha_se:.3f}")
        print(f"- beta SE = {beta_se:.3f}")
        print(f"- gamma SE = {gamma_se:.5f}")

        # T-statistics
        alpha_t = model.tvalues['Intercept']
        beta_t = model.tvalues['oat_f']
        gamma_t = model.tvalues['I(oat_f_shifted * ws_mph)']
        print(f"- alpha t-stat = {alpha_t:.3f}")
        print(f"- beta t-stat = {beta_t:.3f}")
        print(f"- gamma t-stat = {gamma_t:.5f}")

        row = [
            house_alias, month_print, round(alpha,3), round(beta,3), round(gamma,5), 
            round(alpha_se,3), round(beta_se,3), round(gamma_se,5), 
            round(alpha_t,3), round(beta_t,3), round(gamma_t,5)
            ]
        csv_file = 'house_parameters/house_parameters_by_month.csv'
        if not os.path.exists(csv_file):
            pd.DataFrame([row], columns=[
                'house', 'month', 'alpha', 'beta', 'gamma', 
                'alpha_se', 'beta_se', 'gamma_se', 
                'alpha_t', 'beta_t', 'gamma_t'
                ]).to_csv(csv_file, index=False)
        else:
            pd.DataFrame([row], columns=[
                'house', 'month', 'alpha', 'beta', 'gamma', 
                'alpha_se', 'beta_se', 'gamma_se',
                'alpha_t', 'beta_t', 'gamma_t'
                ]).to_csv(csv_file, mode='a', header=False, index=False)

        y_pred = model.predict(test_df)

        rmse = np.sqrt(np.mean((test_df['dist_kwh_scaled'] - y_pred) ** 2))
        print(f"- RMSE: {rmse:.2f} kWh")

        # plt.figure(figsize=(13,4))
        # plt.plot(test_df['hour_start_dt'], test_df['dist_kwh_scaled'], label='Actual distribution scaled', alpha=0.6)
        # plt.plot(test_df['hour_start_dt'], y_pred, label='Predicted distribution scaled', alpha=0.9)
        # plt.ylabel('Heat [kWh]')
        # plt.legend()
        # plt.show()

# Plot for each house
params_df = pd.read_csv('house_parameters/house_parameters_by_month.csv')
oat_range = range(-10,40)
ws = 5

for house in params_df['house'].unique():
    plt.figure(figsize=(8, 4))
    house_data = params_df[params_df['house'] == house]
    
    # Plot a curve for each month with a confidence interval
    for _, row in house_data.iterrows():
        alpha = row['alpha']
        beta = row['beta']
        gamma = row['gamma']
        alpha_se = row['alpha_se']
        beta_se = row['beta_se']
        gamma_se = row['gamma_se']
        month = row['month']
        
        predicted_heat = [alpha + beta*oat + gamma*oat*ws for oat in oat_range]
        
        # Calculate standard error of prediction
        # For each temperature point, calculate the variance of the prediction
        # Var(pred) = Var(alpha) + Var(beta)*oat^2 + Var(gamma)*(oat*ws)^2
        # (assuming parameters are independent)
        pred_se = [np.sqrt(alpha_se**2 + (beta_se*oat)**2 + (gamma_se*oat*ws)**2) for oat in oat_range]
        
        plt.plot(oat_range, predicted_heat, label=f'Month: {month}')
        plt.fill_between(
            oat_range, 
            [h - 1.96*se for h, se in zip(predicted_heat, pred_se)],
            [h + 1.96*se for h, se in zip(predicted_heat, pred_se)],
            alpha=0.2
        )
    
    plt.title(f'House: {house}')
    plt.xlabel('Outside Air Temperature (°F)')
    plt.ylabel("Predicted Heat (kWh) with 95% CI")
    plt.legend()
    plt.tight_layout()
    plt.show()
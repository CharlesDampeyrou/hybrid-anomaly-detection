def show_residue_stats(res_df):
    for flight_name in res_df.get_index_level(0).unique():
        flight_df = res_df.loc[flight_name]
        print(f"Flight {flight_name}")
        print(f"x residue max: {flight_df['x_residue rolling absolute mean'].max()}")
        print(f"y residue max: {flight_df['y_residue rolling absolute mean'].max()}")
        print(f"z residue max: {flight_df['z_residue rolling absolute mean'].max()}")

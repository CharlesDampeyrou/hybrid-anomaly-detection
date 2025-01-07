def add_residue_stats(
    df,
    residue_cols,
    mean_window_size,
    std_window_size,
    ewm_alpha,
):
    new_df = df.copy()
    for col in residue_cols:
        new_df[f"{col} rolling absolute mean"] = (
            df[col].abs().rolling(window=mean_window_size).mean()
        )
        new_df[f"{col} rolling std"] = df[col].rolling(window=std_window_size).std()
        new_df[f"{col} rolling std exponential filtered"] = (
            df[col].rolling(window=std_window_size).std().ewm(alpha=ewm_alpha).mean()
        )
        new_df[f"{col} absolute exponential filtered"] = (
            df[col].abs().ewm(alpha=ewm_alpha).mean()
        )
        new_df[f"{col} exponential filtered absolute"] = (
            df[col].ewm(alpha=ewm_alpha).mean().abs()
        )
    return new_df


def add_anomaly_score(
    df,
    mean_window_size=10,
    std_window_size=10,
    ewm_alpha=0.02,
):
    residue_cols = [col for col in df.columns if "residue" in col]
    new_df = add_residue_stats(
        df,
        residue_cols,
        mean_window_size,
        std_window_size,
        ewm_alpha,
    )
    df["anomaly score"] = 0
    df["anomaly classification"] = 0
    for col in residue_cols:
        df["anomaly score"] += (
            new_df[f"{col} exponential filtered absolute"] / new_df[col].std()
        )
        df["anomaly score"] += (
            new_df[f"{col} rolling std exponential filtered"] / new_df[col].std()
        )
        df["anomaly classification"] += (
            new_df[f"{col} exponential filtered absolute"] / new_df[col].std()
        )
        df["anomaly classification"] -= (
            new_df[f"{col} rolling std exponential filtered"] / new_df[col].std()
        )
    df["anomaly classification"] -= df["anomaly classification"].median()
    return df
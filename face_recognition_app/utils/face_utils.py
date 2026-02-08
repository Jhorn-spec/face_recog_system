def extract_identity(df_list):
    if len(df_list) == 0:
        return None

    df = df_list[0]
    if df.empty:
        return None

    identity_path = df["identity"].values[0]
    file = identity_path.split("/")[-1]
    return file.replace(".jpg", "")

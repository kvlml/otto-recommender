import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")


def get_df_name(df):
    try:
        return df.name
    except AttributeError:
        return None


def plot_user_activity(user, df):
    dict_type_inter_color = dict()
    dict_type_inter_color["clicks"] = "b"
    dict_type_inter_color["carts"] = "orange"
    dict_type_inter_color["orders"] = "r"

    offset = 0.5
    df_cur = df[df["user"] == user]
    df_cur["date_day"] = df_cur["date_day"].astype(str)
    df_name = get_df_name(df)
        
    if df_name == "df_train":
        x = pd.date_range("2022-08-01", "2022-08-28")
    elif df_name == "df_test":
        x = pd.date_range("2022-08-29", "2022-09-04")
    else:
        x = pd.date_range("2022-08-01", "2022-09-04")

    x = [str(d).split(" ")[0] for d in x]
    map_dates = {x[i]: i for i in range(len(x))}
    points_sundays = [point for point in range(len(x)) if (point + 1) % 7 == 0]

    plt.figure(figsize=(12, 7))
    plt.xticks(range(len(x)), x, rotation=90)
    plt.yticks(range(25))

    for point in range(25):
        plt.axhline(y=point, color="gray", linestyle="-", alpha=0.1)
    for point in range(len(x)):
        plt.axvline(x=point + offset, color="gray", linestyle="-", alpha=0.1)
        plt.axvline(x=point - offset, color="gray", linestyle="-", alpha=0.1)
    for point in points_sundays:
        plt.axvline(x=point + offset, color="r", linestyle="-", alpha=0.3)
        plt.axvline(x=point - offset, color="r", linestyle="-", alpha=0.3)
        plt.axvline(x=point - offset - 1, color="r", linestyle="-", alpha=0.3)

    for action, df_cur_action in df_cur.groupby("type"):
        xx = np.random.uniform(-0.15, 0.15, len(df_cur_action))
        yy = np.random.uniform(-0.15, 0.15, len(df_cur_action))
        plt.scatter(
            df_cur_action["date_day"].map(map_dates) + xx,
            df_cur_action["hour"] + yy,
            c=dict_type_inter_color[action],
            s=10,
            label=action,
        )
    plt.legend(
        loc="upper right", bbox_to_anchor=(1.1, 1.0), ncol=1, fancybox=True, shadow=True
    )


def make_date_stats(df):
    df['date'] = pd.to_datetime((df["ts"] + 1000 * 60 * 60 * 2) * 1e6)
    df['date_day'] = df['date'].dt.to_period("d")#.astype(str)
    df['n_day'] = df['date'].dt.day.astype('int8')
    df['n_week'] = df['date'].dt.isocalendar().week.astype('int8')
    df['dayofweek'] = df['date'].dt.dayofweek.astype('int8')
    df['hour'] = df['date'].dt.hour.astype('int8')
    return df

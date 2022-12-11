import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import typing
from beartype import beartype
import warnings

warnings.filterwarnings("ignore")


@beartype
def get_df_name(df: pd.DataFrame) -> typing.Optional[str]:
    try:
        return df.name
    except AttributeError:
        return None


@beartype
def plot_user_activity(list_users: typing.List[int], df: pd.DataFrame) -> None:
    actions = ["clicks", "carts", "orders"]
    dict_type_action = dict()
    for action in actions:
        dict_type_action[action] = dict()

    dict_type_action["clicks"]["color"] = "b"
    dict_type_action["carts"]["color"] = "orange"
    dict_type_action["orders"]["color"] = "r"

    dict_type_action["clicks"]["alpha"] = 0.3
    dict_type_action["carts"]["alpha"] = 0.7
    dict_type_action["orders"]["alpha"] = 0.9

    offset = 0.5
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

    for user in list_users:
        df_cur = df[df["user"] == user]
        df_cur["date_day"] = df_cur["date_day"].astype(str)  
        n_sessions = df_cur['n_session'].max() + 1

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

        dict_count_user_actions = {action: 0 for action in dict_type_action}

        for action in actions:
            df_cur_action = df_cur[df_cur["type"] == action]
            n_actions = len(df_cur_action)
            dict_count_user_actions[action] = n_actions

            xx = np.random.uniform(-0.3, 0.3, n_actions)
            yy = np.random.uniform(-0.3, 0.3, n_actions)

            plt.scatter(
                df_cur_action["date_day"].map(map_dates) + xx,
                df_cur_action["hour"] + yy,
                c=dict_type_action[action]["color"],
                alpha=dict_type_action[action]["alpha"],
                s=10,
                label=action,
            )
        plt.legend(
            loc="upper right", bbox_to_anchor=(1.1, 1.0), ncol=1, fancybox=True, shadow=True
        )

        stats_str = ''
        for action in dict_count_user_actions:
            stats_str += f'{dict_count_user_actions[action]} {action}, '
        stats_str = stats_str[:-2]
        title = f'User {user} has {n_sessions} real sessions with {sum(dict_count_user_actions.values())} total actions: {stats_str}'
        plt.title(title, size=14)
        plt.show()


@beartype
def make_date_stats(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime((df["ts"] + 1000 * 60 * 60 * 2) * 1e6)
    df['date_day'] = df['date'].dt.to_period("d")#.astype(str)
    df['n_day'] = df['date'].dt.day.astype('int8')
    df['n_week'] = df['date'].dt.isocalendar().week.astype('int8')
    df['dayofweek'] = df['date'].dt.dayofweek.astype('int8')
    df['hour'] = df['date'].dt.hour.astype('int8')
    return df

import pandas as pd

from config.config import TOTAL_DAYS

def get_users(actions_df):
    """
    Получить уникальные айди юзеров из таблицы с действиями
    
    Parameters
    ----------
    actions_df : pd.Dataframe
        данные со всеми действиями и сабмитами пользователей
        
    Returns
    -------
    users : pd.Series
    """
    return pd.Series(actions_df.user_id.unique()).to_frame().rename(columns={0: 'user_id'})


def create_interaction(events, submissions):
    """
    Объединить все данные по взаимодействию
    Parameters
    ----------
    events : pd.DataFrame
        данные с действиями пользователей
    submissions : pd.DataFrame
        данные о попытках решения задач
    
    Returns
    -------
    actions : pd.DataFrame
        объединение таблиц, содержащее все действия и попытки решения задач
    """
    actions = pd.concat([events, submissions.rename(columns={'submission_status': 'action'})])
    actions = actions.sort_values(['user_id', 'timestamp', 'action'])
    return actions


def get_target(actions_df, target_action='correct', threshold=40):
    """Посчитать целевую переменную для каждого пользователя.
    Ставится метка 1 в графу passed_course, если пользователь
    произвел действие target_action с >= threshold уникальными степами
    (например, можно поставить correct, чтобы проверять количество решенных задач
    или passed, чтобы проверить количество всех пройденных степов)
    
    Parameters
    ----------
    actions_df : pd.Dataframe
        данные со всеми действиями и сабмитами пользователей
    target_action : str
        название действия, которое считаем
    threshold : int
        количество уникальных степов, нужное для выставления отметки
    
    Returns
    -------
    targets: pd.DataFrame
        таблица с целевыми метками
    """
    user_ids = get_users(actions_df)
    users_count_target = actions_df[
        actions_df.action == target_action
    ].groupby('user_id', as_index=False)[['step_id']].nunique().rename(columns={'step_id': target_action})
    
    # passed_course = 1, если пройдено > threshold           
    users_count_target['passed_course'] = (users_count_target[target_action] >= threshold)
    targets = pd.merge(user_ids, users_count_target, how='left').fillna(0).astype('int').set_index('user_id')
    
    return targets.drop(columns=target_action)


def get_action_counts(actions_df):
    """
    Посчитать количество действий и правильных и неправильных сабмитов
    для каждого пользователя
    
    Parameters
    ----------
    actions_df : pd.Dataframe
        данные со всеми действиями и сабмитами пользователей
    
    Returns
    -------
    users_action_counts: pd.DataFrame
        таблица, содержащая количество взаимдоействий
        discovered, passed, started_attempt, viewed, correct, wrong
        для каждого пользователя за все время, представленное в таблице
    """
    users_action_counts = actions_df.pivot_table(
        index='user_id',
        columns='action',
        values='step_id',
        aggfunc='count',
        fill_value=0
    )
    
    # проверить, что все типы действий есть в таблице
    assert all(pd.Series(['discovered', 'passed', 'started_attempt', 'viewed', 'correct', 'wrong']).\
               isin(actions_df.action.unique()))
    
    users_action_counts = users_action_counts[
         ['discovered', 'passed', 'started_attempt', 'viewed', 'correct', 'wrong']
    ]
    
    return users_action_counts


def get_min_timestamps(actions_df, as_datetime=False):
    """
    Получить минимальный таймстемп действия для каждого пользователя
    
    Parameters
    ----------
    actions_df : pd.Dataframe
        данные со всеми действиями и сабмитами пользователей
    as_datetime : bool
        преобразовывать ли ответ в формат datetime
    
    Returns
    -------
    min_timestamps : pd.DataFrame
    """
    min_timestamps = actions_df.groupby('user_id')[['timestamp']].min()\
        .rename(columns={'timestamp': 'min_timestamp'})
    
    if as_datetime:
        min_timestamps.min_timestamp = pd.to_datetime(min_timestamps.min_timestamp, unit='s')
        
    return min_timestamps


def cut_df_by_time(actions_df, offset=0, hours=24*TOTAL_DAYS):
    """
    Оставить в таблице только данные о действия за промежуток
    от offset до offset + hours часов с начала активности пользователя
    
    Parameters
    ----------
    df: pd.DataFrame
        данные о попытках решения задач
    hours: int
        длина периода в часах, за который мы хотим захватить действия
    offset: int
        длина отступа от начала периода действий каждого пользователя в часах
        
    Returns
    -------
    actions_data_d : pd.DataFrame
    """   
    users_min_and_max_timestamps = get_min_timestamps(actions_df)
    users_min_and_max_timestamps['min_timestamp'] += 60 * 60 * offset
    users_min_and_max_timestamps['max_timestamp'] = users_min_and_max_timestamps['min_timestamp'] + 60 * 60 * hours
    
    actions_data_d = pd.merge(actions_df, users_min_and_max_timestamps, how='inner', on='user_id')
    cond = ((actions_data_d['timestamp'] >= actions_data_d['min_timestamp']) & 
            (actions_data_d['timestamp'] <= actions_data_d['max_timestamp']))
    actions_data_d = actions_data_d[cond]

    assert actions_data_d.user_id.nunique() == actions_df.user_id.nunique()
    return actions_data_d.drop(columns=['min_timestamp', 'max_timestamp'])


def cut_dfs_by_time(events, submissions, days=TOTAL_DAYS):
    """
    Оставить в таблицах events и submissions только данные о действиях,
    которые были осуществлены в течение days суток начиная с первого
    зафиксированного действия пользователя
    
    Parameters
    ----------
    events : pd.DataFrame
        данные с действиями пользователей
    submissions : pd.DataFrame
        данные о попытках решения задач
    days : int
        число дней с начала активности пользователей, за которые мы хотим сохранить данные
        
    Returns
    -------
    events_d, subsmission_d : pd.DataFrame
    """
    min_timestamps =  get_min_timestamps(create_interaction(events, submissions)) 
    # первое действие может быть как ивентом, так и сабмитом
    activity_duration = 60 * 60 * 24 * days
    
    # добавить минимальные таймстемпы и оставить только действия,
    # которые произошли в течение activity_duration секунд 
    # начиная с первого действия
    events_d = events.merge(min_timestamps, how='inner', on='user_id')
    events_d = events_d[events_d.timestamp <= events_d.min_timestamp + activity_duration]
    events_d.drop(columns='min_timestamp', inplace=True)
    assert events_d.user_id.nunique() == events.user_id.nunique()
        
    submissions_d = submissions.merge(min_timestamps, how='inner', on='user_id')
    submissions_d = submissions_d[submissions_d.timestamp <= submissions_d.min_timestamp + activity_duration]
    submissions_d.drop(columns='min_timestamp', inplace=True)
    
    return events_d, submissions_d


def get_action_counts_by_period(actions_df, period_len, total_hours=24*TOTAL_DAYS, sum_only=False, multiindex=False, drop_periods=True):
    """
    Получить разбивку действий пользователя за total_hours часов
    по периодам длиной в period_len часов.
    
    Parameters
    ----------
    actions_df : pd.DataFrame
        данные со всеми действиями и сабмитами пользователей
    total_hours : int
        общая продолжительность активности в часах
    period_len : int
        длина периода активности в часах. должна делить total_hours
    sum_only : bool
        если True, то возвращает только общую сумму количества активностей за периоды
        если False, то возвращает разбивку по всем типам активностей и сабмитов (6 колонок)
    multiindex : bool
        если True и sum_only = True, то возвращает таблицу с мультииндексом (user_id, period)  
        и колонкой action, содержащей суммарное количество действий за период  
    drop_periods : bool
        если False, то возвращает также таблицу со всеми действиями за total_hours часов
        с указанием периода, в который они были совершены
    
    Returns
    -------
    actions_by_period : pd.DataFrame
        таблица, в которой для каждого периода указано количество действий,
        которое было совершено в его течении
    actions_df_cut : pd.DataFrame
        (только если drop_periods=False, см. описание параметра drop_periods)
    """
    
    user_ids = get_users(actions_df)
    
    # рассмотреть только первых total_hours часов активности
    actions_df_cut = cut_df_by_time(actions_df, hours=total_hours)
    
    if total_hours % period_len:
        raise ValueError('Period length must divide total activity duration')
    num_periods = total_hours // period_len
    
    # добавить минимальные таймстемпы для каждого пользователя
    actions_df_cut = actions_df_cut.merge(get_min_timestamps(actions_df), on='user_id')
   
    # для каждого действия добавить номер периода, в котором оно было произведено
    labels = [f'{i * period_len}-{(i + 1) * period_len}H' for i in range(num_periods)]
    actions_df_cut['period'] = pd.cut(actions_df_cut.timestamp - actions_df_cut.min_timestamp,
                                      bins=num_periods, labels=labels)
    actions_df_cut.drop(columns='min_timestamp', inplace=True)
    
    def get_action_counts_during_period(actions_during_period, period_name, user_ids=user_ids, sum_only=sum_only):
        """
        Для таблицы actions_during_period действий во время периода period_name
        посчитать количество действий и сабмитов для каждого пользователя,
        просуммировать все действия (если sum_only = True),
        добавить имя периода в названия колонок
        и добавить возможных недостающих пользователей из таблицы user_ids и приписать им ноль действий за этот период.
        """
        action_counts_during_period = get_action_counts(actions_during_period)
        if sum_only:
            action_counts_during_period = action_counts_during_period.sum(axis=1).to_frame().rename(columns={0: 'actions'})
            
        action_counts_during_period.rename(columns=lambda colname: f'{period_name}_{colname}', inplace=True)
        
        action_counts_during_period = user_ids.merge(action_counts_during_period, on='user_id', how='left').\
            fillna(0).astype(int).set_index('user_id')
        return action_counts_during_period
    
    # собрать действия по периодам в одну таблицу
    actions_by_period = pd.concat([
        get_action_counts_during_period(actions_during_period, period_name)
        for period_name, actions_during_period in actions_df_cut.groupby('period')
    ], axis=1)
    
    if sum_only and multiindex:
        # собрать одноколоночную мультииндексную таблицу
        actions_by_period = pd.DataFrame(
            actions_by_period.values.flatten(),
            index=pd.MultiIndex.from_product([actions_by_period.index, actions_by_period.columns])
        ).rename(columns={0: 'action'})
        
    if drop_periods:
        return actions_by_period
    
    # вернуть также таблицу с указанием действия и периодов, в которые они были совершены
    return actions_by_period, actions_df_cut


def get_base_features(actions_df, add_num_days=True, target=None):
    """
    Посчитать базовые признаки для задачи классификации
    
    Parameters
    ----------
    actions_df : pd.DataFrame
        данные со всеми действиями и сабмитами пользователей
    add_num_days : bool
        добавлять ли количество уникальных дней, проведенных пользователем на курсе
    target : pd.DataFrame
        таблица с целевыми метками для пользователей
    
    Returns
    -------
    users_action_counts : pd.DataFrame
        таблица с базовыми признаками пользователей
    """
    
    # посчитать число действий каждого пользователя
    base_features = get_action_counts(actions_df)
    
    if add_num_days:
        # добавить число уникальных дней на курсе
        actions_df['date'] = actions_df.time.dt.date
        num_days = actions_df.groupby('user_id')[['date']].nunique().rename(columns={'date': 'num_days'})
        base_features = base_features.merge(num_days, on='user_id', how='outer').fillna(0).astype(int)
        actions_df.drop(columns='date', inplace=True)

        assert base_features.index.nunique() == actions_df.user_id.nunique()
    
    if target is not None:
        # добавление целевой переменной
        base_features = base_features.merge(target, on='user_id', how='left').\
            fillna(0).astype(int)

    return base_features
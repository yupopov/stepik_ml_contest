import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_validate
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from config.config import RANDOM_STATE


def test_model(model, X, y, fit_params=dict(), test_size=0.1, cv=10, random_state=RANDOM_STATE, figsize=None):
  """
  Разбить множество (X, y) на тренировочную и валидационную части,
  обучить модель model на тренировочной, получить оценки качества (ROC-AUC).
  Провести кросс-валидацию, получить среднее и разброс оценок.
  Если у модели есть аттрибут feature_importances_,
  построить boxplot значимостей признаков на кросс-валидации.

  Parameters
  ----------
  model : estimator instance
    Модель, которую тестируем. Должна иметь методы fit и predict_proba

  X : array-like of shape (n_samples, n_features)
    Training vector, where ``n_samples`` is the number of samples and
    ``n_features`` is the number of features.

  y : array-like of shape (n_samples) or (n_samples, n_features)
    Target relative to ``X`` for classification or regression;

  fit_params : dict
    Параметры, с которыми обучаем модель

  test_size : float in range (0, 1)
    Доля тестовой выборки

  random_state : nt, RandomState instance or None, default=None
    Контролирует `random_state` модели и разбиения выборки
  """
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

  model_ = clone(model)
  model_.random_state = random_state

  if model_.__class__.__name__ == 'XGBClassifier':
    # добавить валидационное множество в случае бустинга
    eval_set = [(X_valid, y_valid)]
    fit_params_ = fit_params.copy()
    fit_params_['eval_set'] = eval_set
    model_.fit(X_train, y_train, **fit_params_)
  else:
    model_.fit(X_train, y_train)

  roc_train = roc_auc_score(y_train, model_.predict_proba(X_train)[:, 1])
  roc_valid = roc_auc_score(y_valid, model_.predict_proba(X_valid)[:, 1])

  print(f'ROC-AUC на тренировочной выборке:\t {roc_train}')
  print(f'ROC-AUC на валидационной выборке:\t {roc_valid}\n')

  # посчитать оценки на кросс-валидации
  cv_results = cross_validate(model_, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, return_estimator=True)

  cv_scores = cv_results['test_score']
  cv_mean = cv_scores.mean()
  cv_std = cv_scores.std()

  print(f'Mean CV score: {cv_mean}')
  print(f'Std of CV scores: {cv_std}')

  if hasattr(model_, 'feature_importances_'):
    # построить boxplot значимости фичей на кросс-валидации
    estimators = cv_results['estimator']
    feature_importances = pd.DataFrame(
        np.vstack([estimator.feature_importances_ for estimator in estimators]),
        columns=X_train.columns
        ).T

    sorted_idx = feature_importances.mean(axis=1).sort_values().index
    feature_importances = feature_importances.loc[sorted_idx, :]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(feature_importances,
            vert=False, labels=feature_importances.index)
    ax.set_title("Feature importances on CV")
    fig.tight_layout()


def plot_learning_curve(estimator, X, y, title=None, scoring=None, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    title : str
        Title for the chart.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or a scorer callable
        object / function with signature scorer(estimator, X, y).

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    if title is None:
      title = f'Learning curves ({estimator.__class__.__name__})'

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")


def plot_validation_curve(estimator, X, y, param_name, param_range, plot=plt.plot, title=None, scoring=None, ylim=None, cv=None,
                        n_jobs=None):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    plot : function, default=plt.plot
        Type of the plot for the validation curve.
        Could be `plt.semilogx` for a logarithmic param_range, for example.

    title : str, default=None
        Title for the chart.
        If None, prints Validation curve for `estimator.__class__.__name__` with respect to `param_name`.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or a scorer callable
        object / function with signature `scorer(estimator, X, y)`.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    train_scores, test_scores = validation_curve(
        estimator, X, y,
        param_name=param_name, param_range=param_range,
        scoring=scoring, cv=cv, n_jobs=n_jobs
        )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if title is None:
      title = f'Validation curve for {estimator.__class__.__name__} with respect to {param_name}'
    plt.title(title)
    plt.grid()
    plt.xlabel(param_name)
    plt.xticks(ticks=param_range)
    plt.ylabel(str(scoring))
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    
    plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    
    plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    
    plt.legend(loc="best")
    plt.show()

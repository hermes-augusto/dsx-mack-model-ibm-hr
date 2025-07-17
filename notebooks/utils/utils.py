import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  RandomizedSearchCV
from skopt import BayesSearchCV
import optuna
from sklearn.model_selection import cross_val_score


def generate_ibm_hr_dataset(n_samples=1_000_000, seed=42):
    """
    Gera um dataset sintético inspirado no IBM HR Analytics com relações mais realistas

    Parameters:
    n_samples (int): Número de amostras a gerar
    seed (int): Seed para reprodutibilidade

    Returns:
    pd.DataFrame: Dataset gerado
    """
    np.random.seed(seed)

    # Gerando dados base
    data = {}

    # Idade com distribuição mais realista (concentrada entre 25-50 anos)
    data['Age'] = np.random.normal(38, 10, n_samples).astype(int)
    data['Age'] = np.clip(data['Age'], 18, 65)

    # Gênero
    data['Gender'] = np.random.choice(['Female', 'Male'], n_samples, p=[0.4, 0.6])

    # Educação (1-5: Below College, College, Bachelor, Master, Doctor)
    education_probs = [0.05, 0.15, 0.40, 0.30, 0.10]
    data['Education'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=education_probs)

    # Campo educacional baseado no nível de educação
    education_fields = ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']
    data['EducationField'] = np.random.choice(education_fields, n_samples)

    # Departamento
    dept_probs = [0.45, 0.40, 0.15]  # Sales mais comum, HR menos comum
    data['Department'] = np.random.choice(['Sales', 'Research & Development', 'Human Resources'],
                                        n_samples, p=dept_probs)

    # Job Role baseado no departamento
    job_roles = {
        'Sales': ['Sales Executive', 'Sales Representative', 'Manager'],
        'Research & Development': ['Research Scientist', 'Laboratory Technician', 'Research Director', 'Manager'],
        'Human Resources': ['Human Resources', 'Manager']
    }

    data['JobRole'] = np.empty(n_samples, dtype=object)
    for dept in job_roles:
        mask = np.array(data['Department']) == dept
        n_dept = mask.sum()
        if n_dept > 0:
            data['JobRole'][mask] = np.random.choice(job_roles[dept], n_dept)

    # Job Level correlacionado com idade e educação
    base_level = np.ones(n_samples)
    age_bonus = (np.array(data['Age']) - 18) / 47 * 2  # 0-2 pontos baseado na idade
    edu_bonus = (np.array(data['Education']) - 1) / 4 * 2  # 0-2 pontos baseado na educação

    data['JobLevel'] = np.round(base_level + age_bonus + edu_bonus).astype(int)
    data['JobLevel'] = np.clip(data['JobLevel'], 1, 5)

    # Total Working Years correlacionado com idade
    data['TotalWorkingYears'] = np.maximum(0, data['Age'] - 18 - np.random.randint(0, 5, n_samples))

    # Years at Company (não pode ser maior que TotalWorkingYears)
    data['YearsAtCompany'] = np.random.randint(0, 21, n_samples)
    data['YearsAtCompany'] = np.minimum(data['YearsAtCompany'], data['TotalWorkingYears'])

    # Years in Current Role (não pode ser maior que YearsAtCompany)
    data['YearsInCurrentRole'] = np.random.randint(0, 11, n_samples)
    data['YearsInCurrentRole'] = np.minimum(data['YearsInCurrentRole'], data['YearsAtCompany'])

    # Years Since Last Promotion
    data['YearsSinceLastPromotion'] = np.random.randint(0, 8, n_samples)
    data['YearsSinceLastPromotion'] = np.minimum(data['YearsSinceLastPromotion'], data['YearsAtCompany'])

    # Years With Current Manager
    data['YearsWithCurrManager'] = np.random.randint(0, 8, n_samples)
    data['YearsWithCurrManager'] = np.minimum(data['YearsWithCurrManager'], data['YearsInCurrentRole'])

    # Número de empresas trabalhadas (correlacionado com anos totais de trabalho)
    max_companies = np.minimum(data['TotalWorkingYears'] // 2, 9)
    data['NumCompaniesWorked'] = np.array([np.random.randint(0, max(1, mc) + 1) for mc in max_companies])

    # Monthly Income correlacionado com JobLevel, Education e TotalWorkingYears
    base_income = 2000
    level_factor = data['JobLevel'] * 2000
    education_factor = data['Education'] * 500
    experience_factor = data['TotalWorkingYears'] * 100
    noise = np.random.normal(0, 1000, n_samples)

    data['MonthlyIncome'] = base_income + level_factor + education_factor + experience_factor + noise
    data['MonthlyIncome'] = np.clip(data['MonthlyIncome'].astype(int), 1000, 20000)

    # Rates
    data['DailyRate'] = np.random.randint(100, 1500, n_samples)
    data['HourlyRate'] = np.random.randint(30, 100, n_samples)
    data['MonthlyRate'] = np.random.randint(2000, 27000, n_samples)

    # Distance from home (distribuição exponencial - mais pessoas moram perto)
    data['DistanceFromHome'] = np.random.exponential(7, n_samples).astype(int) + 1
    data['DistanceFromHome'] = np.clip(data['DistanceFromHome'], 1, 29)

    # Business Travel
    travel_probs = [0.70, 0.20, 0.10]  # Maioria viaja raramente
    data['BusinessTravel'] = np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
                                            n_samples, p=travel_probs)

    # Satisfação e envolvimento
    data['EnvironmentSatisfaction'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.10, 0.20, 0.40, 0.30])
    data['JobSatisfaction'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.10, 0.20, 0.40, 0.30])
    data['RelationshipSatisfaction'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.10, 0.20, 0.40, 0.30])
    data['JobInvolvement'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.05, 0.15, 0.50, 0.30])
    data['WorkLifeBalance'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.10, 0.25, 0.45, 0.20])

    # Performance Rating (maioria boa performance)
    data['PerformanceRating'] = np.random.choice([3, 4], n_samples, p=[0.84, 0.16])

    # Percent Salary Hike correlacionado com Performance Rating
    data['PercentSalaryHike'] = np.where(
        data['PerformanceRating'] == 4,
        np.random.randint(15, 26, n_samples),
        np.random.randint(11, 18, n_samples)
    )

    # Stock Option Level correlacionado com JobLevel
    data['StockOptionLevel'] = np.random.choice([0, 1, 2, 3], n_samples,
                                               p=[0.40, 0.35, 0.20, 0.05])
    high_level_mask = data['JobLevel'] >= 4
    data['StockOptionLevel'][high_level_mask] = np.random.choice([1, 2, 3],
                                                                 high_level_mask.sum(),
                                                                 p=[0.30, 0.50, 0.20])

    # Training Times Last Year
    data['TrainingTimesLastYear'] = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples,
                                                    p=[0.05, 0.10, 0.25, 0.30, 0.20, 0.08, 0.02])

    # Marital Status
    data['MaritalStatus'] = np.random.choice(['Single', 'Married', 'Divorced'], n_samples,
                                           p=[0.32, 0.55, 0.13])

    # OverTime - maior probabilidade para níveis menores e pessoas mais jovens
    overtime_base_prob = 0.28
    age_factor = (65 - data['Age']) / 47 * 0.1  # Jovens trabalham mais overtime
    level_factor = (5 - data['JobLevel']) / 4 * 0.1  # Níveis menores trabalham mais overtime

    overtime_prob = np.clip(overtime_base_prob + age_factor + level_factor, 0.1, 0.5)
    data['OverTime'] = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in overtime_prob]

    # Attrition - baseado em múltiplos fatores
    attrition_score = np.zeros(n_samples)

    # Fatores que aumentam attrition
    attrition_score += (data['JobSatisfaction'] == 1) * 0.15
    attrition_score += (data['EnvironmentSatisfaction'] == 1) * 0.10
    attrition_score += (data['WorkLifeBalance'] == 1) * 0.10
    attrition_score += (np.array(data['OverTime']) == 'Yes') * 0.08
    attrition_score += (data['YearsSinceLastPromotion'] > 5) * 0.05
    attrition_score += (data['DistanceFromHome'] > 20) * 0.05
    attrition_score += (np.array(data['MaritalStatus']) == 'Single') * 0.03
    attrition_score += (data['NumCompaniesWorked'] > 5) * 0.04

    # Fatores que diminuem attrition
    attrition_score -= (data['JobLevel'] >= 4) * 0.10
    attrition_score -= (data['YearsAtCompany'] > 10) * 0.08
    attrition_score -= (data['StockOptionLevel'] > 0) * 0.05

    # Probabilidade base de 16%
    attrition_prob = np.clip(0.16 + attrition_score, 0.05, 0.50)
    data['Attrition'] = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in attrition_prob]

    # Campos fixos
    data['EmployeeCount'] = np.ones(n_samples, dtype=int)
    data['EmployeeNumber'] = np.arange(1, n_samples + 1)
    data['Over18'] = ['Y'] * n_samples
    data['StandardHours'] = [80] * n_samples

    # Criar DataFrame
    df = pd.DataFrame(data)

    # Reordenar colunas para match com o dataset original
    column_order = [
        'Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
        'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
        'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]

    return df[column_order]


def tune_model(model, search_type, param_grid, X, y, n_iter=20, scoring='f1'):
    """
    search_type: 'grid', 'random', 'bayes', 'optuna'
    """
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
        search.fit(X, y)
        best = search.best_estimator_
        best_score = search.best_score_
        best_params = search.best_params_
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=3, scoring=scoring, n_jobs=-1, random_state=42)
        search.fit(X, y)
        best = search.best_estimator_
        best_score = search.best_score_
        best_params = search.best_params_
    elif search_type == 'bayes':
        search = BayesSearchCV(model, param_grid, n_iter=n_iter, cv=3, scoring=scoring, n_jobs=-1, random_state=42)
        search.fit(X, y)
        best = search.best_estimator_
        best_score = search.best_score_
        best_params = search.best_params_
    elif search_type == 'optuna':
        def objective(trial):
            param_suggest = {}
            for param, space in param_grid.items():
                if isinstance(space, tuple) and len(space) == 3 and space[2] == 'log-uniform':
                    param_suggest[param] = trial.suggest_loguniform(param, space[0], space[1])
                elif isinstance(space, tuple) and len(space) == 2 and isinstance(space[0], float):
                    param_suggest[param] = trial.suggest_uniform(param, space[0], space[1])
                elif isinstance(space, tuple) and len(space) == 2 and isinstance(space[0], int):
                    param_suggest[param] = trial.suggest_int(param, space[0], space[1])
                elif isinstance(space, list):
                    param_suggest[param] = trial.suggest_categorical(param, space)
                else:
                    raise ValueError("Tipo de espaço de busca não suportado.")
            model.set_params(**param_suggest)
            return cross_val_score(model, X, y, cv=3, scoring=scoring).mean()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_iter)
        best_params = study.best_params
        model.set_params(**best_params)
        best = model.fit(X, y)
        best_score = cross_val_score(model, X, y, cv=3, scoring=scoring).mean()
    else:
        raise ValueError('Tipo de busca não reconhecido!')
    return best, best_params, best_score
    
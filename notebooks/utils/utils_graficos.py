import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve,precision_score,recall_score, f1_score


def plot_features_grid(df, features_novas):
    n_cols = 3
    box_features_list = [f for f in features_novas if f not in ['CargoAlto_HorasExtras','EstavelNaEmpresa','Burnout']]
    countplot_features_list = [f for f in features_novas if f in ['CargoAlto_HorasExtras','EstavelNaEmpresa','Burnout']]
    all_features = box_features_list + countplot_features_list
    total = len(all_features)
    n_rows = math.ceil(total / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
    axes = axes.flatten()

    for idx, feature in enumerate(all_features):
        ax = axes[idx]
        if feature in box_features_list:
            sns.boxplot(data=df, x='Attrition', y=feature, ax=ax)
            ax.set_title(f'Boxplot de {feature} por Attrition')
            ax.set_xlabel('Attrition')
            ax.set_ylabel(feature)
        else:
            sns.countplot(data=df, x=feature, hue='Attrition', ax=ax)
            ax.set_title(f'Distribuição de {feature} por Attrition')
            ax.set_xlabel(feature)
            ax.set_ylabel('Número de Funcionários')

    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def avaliar_modelo(modelo, X_train, X_test, y_train, y_test, nome='Modelo'):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X_test)

    print(f"\n{nome}\n", "="*40)
    print(classification_report(y_test, y_pred, digits=3))
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Matriz de Confusão - {nome}')
    axes[0].set_xlabel('Predito')
    axes[0].set_ylabel('Real')

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    sns.lineplot(x=fpr, y=tpr, ax=axes[1],  label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0,1],[0,1],'k--')
    axes[1].set_title(f'Curva ROC - {nome}')
    axes[1].set_xlabel('Falso Positivo')
    axes[1].set_ylabel('Verdadeiro Positivo')
    axes[1].legend()
    
    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    sns.lineplot(x=recall, y=precision, ax=axes[2])
    axes[2].set_title(f'Precision-Recall Curve - {nome}')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    
    # Precision, Recall e F1-score vs Threshold
    thresholds = np.arange(0.0, 1.01, 0.05)
    precisions, recalls, f1s = [], [], []
    for thresh in thresholds:
        y_pred_th = (y_proba >= thresh).astype(int)
        precisions.append(precision_score(y_test, y_pred_th, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_th, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_th, zero_division=0))

    axes[3].plot(thresholds, precisions, label='Precision')
    axes[3].plot(thresholds, recalls, label='Recall')
    axes[3].plot(thresholds, f1s, label='F1-score')
    axes[3].set_xlabel('Threshold')
    axes[3].set_ylabel('Score')
    axes[3].set_title(f'Prec/Recall/F1 vs Threshold')
    axes[3].legend()
    axes[3].grid()

    plt.tight_layout()
    plt.show()
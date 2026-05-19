# =============================================================================
# PROYECTO FINAL - SISTEMAS INTELIGENTES
# PREDICCIÓN DE VENTAS DE PRODUCTOS
# Universidad Católica de Santa María - Ingeniería de Sistemas
# Dataset: statsfinal.csv (4,600 registros reales)
# Algoritmos: Regresión Logística, KNN, SVM, Árbol de Decisión
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGA Y LIMPIEZA DEL DATASET REAL
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
local_path = os.path.join(script_dir, 'statsfinal.csv')
alt_path = os.path.join(script_dir, 'data', 'statsfinal.csv')

if os.path.exists(local_path):
    dataset_path = local_path
elif os.path.exists(alt_path):
    dataset_path = alt_path
else:
    raise FileNotFoundError(
        'Dataset no encontrado. Coloca statsfinal.csv en la carpeta del proyecto ' 
        f'({script_dir}) o en {os.path.join(script_dir, "data")}.')

print(f'Usando dataset: {dataset_path}')
df = pd.read_csv(dataset_path)
df.drop(columns=['Unnamed: 0'], inplace=True)

output_dir = os.path.join(script_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Parsear fecha robustamente
def parse_date(s):
    for fmt in ('%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d'):
        try:
            return pd.to_datetime(s, format=fmt)
        except:
            pass
    return pd.NaT

df['Date'] = df['Date'].apply(parse_date)
df.dropna(subset=['Date'], inplace=True)

print("=" * 62)
print("  PROYECTO: PREDICCIÓN DE VENTAS DE PRODUCTOS")
print("  Universidad Católica de Santa María - SI")
print("=" * 62)
print(f"\n[1] DATASET CARGADO: {len(df)} registros reales")
print(f"    Período: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"    Columnas originales: {list(df.columns)}")

# =============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================

# Variables temporales
df['mes']           = df['Date'].dt.month
df['dia_semana']    = df['Date'].dt.dayofweek   # 0=Lun, 6=Dom
df['trimestre']     = df['Date'].dt.quarter
df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)

# Totales agregados por día
df['venta_total']    = df['S-P1'] + df['S-P2'] + df['S-P3'] + df['S-P4']
df['cantidad_total'] = df['Q-P1'] + df['Q-P2'] + df['Q-P3'] + df['Q-P4']

# Precio unitario implícito por producto
df['precio_P1'] = df['S-P1'] / df['Q-P1']
df['precio_P2'] = df['S-P2'] / df['Q-P2']
df['precio_P3'] = df['S-P3'] / df['Q-P3']
df['precio_P4'] = df['S-P4'] / df['Q-P4']

# Participación de cada producto en la venta total del día
df['part_P1'] = df['S-P1'] / df['venta_total']
df['part_P2'] = df['S-P2'] / df['venta_total']
df['part_P3'] = df['S-P3'] / df['venta_total']
df['part_P4'] = df['S-P4'] / df['venta_total']

# Variable objetivo: venta_alta = 1 si venta_total > mediana
mediana = df['venta_total'].median()
df['venta_alta'] = (df['venta_total'] > mediana).astype(int)

print(f"\n[2] INGENIERÍA DE CARACTERÍSTICAS:")
print(f"    Variables temporales: mes, dia_semana, trimestre, es_fin_semana")
print(f"    Variables agregadas: venta_total, cantidad_total, precios, participaciones")
print(f"    Umbral venta_alta: S/ {mediana:,.2f}")
print(f"    Distribución clase 0 (Baja): {(df['venta_alta']==0).sum()}")
print(f"    Distribución clase 1 (Alta): {(df['venta_alta']==1).sum()}")

# =============================================================================
# 3. PREPROCESAMIENTO
# =============================================================================

features = [
    'Q-P1', 'Q-P2', 'Q-P3', 'Q-P4',
    'S-P1', 'S-P2', 'S-P3', 'S-P4',
    'precio_P1', 'precio_P2', 'precio_P3', 'precio_P4',
    'part_P1', 'part_P2', 'part_P3', 'part_P4',
    'mes', 'dia_semana', 'trimestre', 'es_fin_semana',
    'cantidad_total'
]

X = df[features]
y = df['venta_alta']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\n[3] PREPROCESAMIENTO:")
print(f"    Features: {len(features)} variables")
print(f"    Train: {len(X_train)} registros | Test: {len(X_test)} registros")

# =============================================================================
# 4. ENTRENAMIENTO DE MODELOS
# =============================================================================

resultados = {}

# ── 4.1 REGRESIÓN LOGÍSTICA ─────────────────────────────────────────────────
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=2000, random_state=42, C=1.0))
])
pipe_lr.fit(X_train, y_train)
y_pred_lr  = pipe_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, pipe_lr.predict_proba(X_test)[:, 1])
resultados['Regresión Logística'] = {
    'acc': acc_lr, 'auc': auc_lr, 'y_pred': y_pred_lr, 'modelo': pipe_lr}
print(f"\n[4.1] REGRESIÓN LOGÍSTICA  → Accuracy: {acc_lr:.4f} | AUC: {auc_lr:.4f}")

# ── 4.2 KNN ─────────────────────────────────────────────────────────────────
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  KNeighborsClassifier(n_neighbors=7, metric='euclidean'))
])
pipe_knn.fit(X_train, y_train)
y_pred_knn = pipe_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
auc_knn = roc_auc_score(y_test, pipe_knn.predict_proba(X_test)[:, 1])
resultados['KNN (k=7)'] = {
    'acc': acc_knn, 'auc': auc_knn, 'y_pred': y_pred_knn, 'modelo': pipe_knn}
print(f"[4.2] KNN (k=7)            → Accuracy: {acc_knn:.4f} | AUC: {auc_knn:.4f}")

# ── 4.3 SVM ─────────────────────────────────────────────────────────────────
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  SVC(kernel='rbf', probability=True, random_state=42, C=1.0))
])
pipe_svm.fit(X_train, y_train)
y_pred_svm = pipe_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, pipe_svm.predict_proba(X_test)[:, 1])
resultados['SVM (RBF)'] = {
    'acc': acc_svm, 'auc': auc_svm, 'y_pred': y_pred_svm, 'modelo': pipe_svm}
print(f"[4.3] SVM (kernel RBF)     → Accuracy: {acc_svm:.4f} | AUC: {auc_svm:.4f}")

# ── 4.4 ÁRBOL DE DECISIÓN ───────────────────────────────────────────────────
dt = DecisionTreeClassifier(max_depth=6, random_state=42, min_samples_leaf=10)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
resultados['Árbol de Decisión'] = {
    'acc': acc_dt, 'auc': auc_dt, 'y_pred': y_pred_dt, 'modelo': dt}
print(f"[4.4] ÁRBOL DE DECISIÓN    → Accuracy: {acc_dt:.4f} | AUC: {auc_dt:.4f}")

# =============================================================================
# 5. VISUALIZACIONES
# =============================================================================

nombres = list(resultados.keys())
accs    = [resultados[m]['acc'] for m in nombres]
aucs    = [resultados[m]['auc'] for m in nombres]
colores = ['#1F3864', '#2E5BAA', '#C00000', '#375623']

# ── Figura 1: Comparación de modelos ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    'Proyecto Final SI – Predicción de Ventas de Productos\n'
    'Dataset: statsfinal.csv (4,600 registros) | UCSM',
    fontsize=13, fontweight='bold', color='#1F3864')

# Accuracy
ax = axes[0, 0]
bars = ax.bar(nombres, accs, color=colores, edgecolor='white', linewidth=1.3)
ax.set_ylim(0, 1.15)
ax.set_title('Accuracy por Algoritmo', fontweight='bold', color='#1F3864')
ax.set_ylabel('Accuracy')
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticklabels(nombres, rotation=15, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# AUC
ax = axes[0, 1]
bars2 = ax.bar(nombres, aucs, color=colores, edgecolor='white', linewidth=1.3)
ax.set_ylim(0, 1.15)
ax.set_title('AUC-ROC por Algoritmo', fontweight='bold', color='#1F3864')
ax.set_ylabel('AUC-ROC')
for bar, val in zip(bars2, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticklabels(nombres, rotation=15, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Curvas ROC
ax = axes[0, 2]
for nombre, color in zip(nombres, colores):
    mod = resultados[nombre]['modelo']
    proba = mod.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{nombre} ({resultados[nombre]["auc"]:.3f})')
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set_title('Curvas ROC', fontweight='bold', color='#1F3864')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# Matrices de confusión (3 mejores)
for idx, (nombre, ax) in enumerate(zip(nombres[:3], axes[1])):
    cm = confusion_matrix(y_test, resultados[nombre]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Baja','Alta'], yticklabels=['Baja','Alta'],
                annot_kws={'size': 13, 'weight': 'bold'})
    acc_val = resultados[nombre]['acc']
    ax.set_title(f'{nombre}\nAcc={acc_val:.3f}',
                 fontweight='bold', color='#1F3864', fontsize=9)
    ax.set_xlabel('Predicho'); ax.set_ylabel('Real')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparacion_modelos.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[5] Figura 1 guardada: {os.path.join(output_dir, 'comparacion_modelos.png')}")

# ── Figura 2: EDA del dataset real ─────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
fig2.suptitle('Análisis Exploratorio – Dataset statsfinal.csv (4,600 registros)',
              fontweight='bold', color='#1F3864', fontsize=13)

# Ventas por mes
ventas_mes = df.groupby('mes')['venta_total'].mean()
axes2[0,0].bar(ventas_mes.index, ventas_mes.values, color='#2E5BAA', edgecolor='white')
axes2[0,0].set_title('Venta Total Promedio por Mes', fontweight='bold', color='#1F3864')
axes2[0,0].set_xlabel('Mes'); axes2[0,0].set_ylabel('S/ Promedio')
axes2[0,0].set_xticks(range(1,13))
axes2[0,0].grid(axis='y', alpha=0.3)

# Ventas por día de semana
dias = ['Lun','Mar','Mié','Jue','Vie','Sáb','Dom']
ventas_dia = df.groupby('dia_semana')['venta_total'].mean()
axes2[0,1].bar(range(len(ventas_dia)), ventas_dia.values, color='#1F3864', edgecolor='white')
axes2[0,1].set_xticks(range(7)); axes2[0,1].set_xticklabels(dias)
axes2[0,1].set_title('Venta Total Promedio por Día de Semana', fontweight='bold', color='#1F3864')
axes2[0,1].set_ylabel('S/ Promedio'); axes2[0,1].grid(axis='y', alpha=0.3)

# Distribución venta_total (histograma)
axes2[1,0].hist(df['venta_total'], bins=50, color='#2E5BAA', edgecolor='white', alpha=0.85)
axes2[1,0].axvline(mediana, color='#C00000', lw=2, linestyle='--', label=f'Mediana: S/ {mediana:,.0f}')
axes2[1,0].set_title('Distribución de Venta Total Diaria', fontweight='bold', color='#1F3864')
axes2[1,0].set_xlabel('Venta Total (S/)'); axes2[1,0].set_ylabel('Frecuencia')
axes2[1,0].legend(); axes2[1,0].grid(axis='y', alpha=0.3)

# Ventas por producto (boxplot)
ventas_prod = df[['S-P1','S-P2','S-P3','S-P4']]
bp = axes2[1,1].boxplot(ventas_prod.values, labels=['P1','P2','P3','P4'],
                         patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], colores):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes2[1,1].set_title('Distribución de Ventas por Producto', fontweight='bold', color='#1F3864')
axes2[1,1].set_xlabel('Producto'); axes2[1,1].set_ylabel('Ventas (S/)')
axes2[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_dataset.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"[5] Figura 2 guardada: {os.path.join(output_dir, 'eda_dataset.png')}")

# ── Figura 3: Serie temporal de ventas ──────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(15, 4))
df_sorted = df.sort_values('Date')
ax3.plot(df_sorted['Date'], df_sorted['venta_total'],
         color='#2E5BAA', lw=0.7, alpha=0.8)
ax3.axhline(mediana, color='#C00000', lw=1.5, linestyle='--',
            label=f'Umbral Venta Alta: S/ {mediana:,.0f}')
ax3.fill_between(df_sorted['Date'], df_sorted['venta_total'], mediana,
                 where=df_sorted['venta_total'] > mediana,
                 alpha=0.25, color='#375623', label='Venta Alta')
ax3.fill_between(df_sorted['Date'], df_sorted['venta_total'], mediana,
                 where=df_sorted['venta_total'] <= mediana,
                 alpha=0.25, color='#C00000', label='Venta Baja')
ax3.set_title('Serie Temporal de Venta Total Diaria – statsfinal.csv',
              fontweight='bold', color='#1F3864', fontsize=12)
ax3.set_xlabel('Fecha'); ax3.set_ylabel('Venta Total (S/)')
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'serie_temporal.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"[5] Figura 3 guardada: {os.path.join(output_dir, 'serie_temporal.png')}")

# ── Figura 4: Importancia de variables (Árbol de Decisión) ──────────────────
importancias = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=True)
top15 = importancias.tail(15)
fig4, ax4 = plt.subplots(figsize=(10, 6))
colors_bar = ['#C00000' if v > importancias.median() else '#2E5BAA' for v in top15.values]
ax4.barh(top15.index, top15.values, color=colors_bar, edgecolor='white')
ax4.set_title('Top 15 Variables más Importantes – Árbol de Decisión',
              fontweight='bold', color='#1F3864', fontsize=11)
ax4.set_xlabel('Importancia (Gini)'); ax4.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'importancia_variables.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"[5] Figura 4 guardada: {os.path.join(output_dir, 'importancia_variables.png')}")

# =============================================================================
# 6. INTERFAZ DE USUARIO CON VALIDACIONES
# =============================================================================

def validar_float(val, nombre, minv, maxv):
    try:
        v = float(val)
        if v < minv or v > maxv:
            raise ValueError(f"{nombre} debe estar entre {minv} y {maxv}")
        return v
    except (TypeError, ValueError) as e:
        raise ValueError(str(e))

def validar_int(val, nombre, minv, maxv):
    try:
        v = int(float(val))
        if v < minv or v > maxv:
            raise ValueError(f"{nombre} debe ser entero entre {minv} y {maxv}")
        return v
    except (TypeError, ValueError) as e:
        raise ValueError(str(e))

def predecir_venta_real(q1, q2, q3, q4, s1, s2, s3, s4,
                        mes, dia_semana, modelo_nombre='SVM (RBF)'):
    """Predice si el día tendrá venta alta o baja."""
    trimestre   = (mes - 1) // 3 + 1
    fin_semana  = 1 if dia_semana >= 5 else 0
    p1 = s1 / q1 if q1 > 0 else 0
    p2 = s2 / q2 if q2 > 0 else 0
    p3 = s3 / q3 if q3 > 0 else 0
    p4 = s4 / q4 if q4 > 0 else 0
    vt = s1 + s2 + s3 + s4
    ct = q1 + q2 + q3 + q4
    part1 = s1/vt if vt > 0 else 0
    part2 = s2/vt if vt > 0 else 0
    part3 = s3/vt if vt > 0 else 0
    part4 = s4/vt if vt > 0 else 0

    X_nuevo = np.array([[q1, q2, q3, q4, s1, s2, s3, s4,
                          p1, p2, p3, p4,
                          part1, part2, part3, part4,
                          mes, dia_semana, trimestre, fin_semana, ct]])

    modelo = resultados[modelo_nombre]['modelo']
    pred   = modelo.predict(X_nuevo)[0]
    prob   = modelo.predict_proba(X_nuevo)[0]
    return {
        'prediccion':       'VENTA ALTA ▲' if pred == 1 else 'VENTA BAJA ▼',
        'confianza':        f'{max(prob)*100:.1f}%',
        'venta_total_est':  f'S/ {vt:,.2f}',
        'vs_umbral':        f'S/ {vt - mediana:+,.2f} respecto al umbral',
        'algoritmo':        modelo_nombre
    }

def interfaz_consola():
    dias_sem = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
    print("\n" + "=" * 62)
    print("  SISTEMA DE PREDICCIÓN DE VENTAS DE PRODUCTOS")
    print("  Universidad Católica de Santa María – SI")
    print("=" * 62)
    print("  Ingrese los datos del día a predecir:\n")

    campos = [
        ("Cantidad vendida Producto 1 (Q-P1)",  lambda v: validar_int(v,   "Q-P1",   1, 10000)),
        ("Cantidad vendida Producto 2 (Q-P2)",  lambda v: validar_int(v,   "Q-P2",   1, 10000)),
        ("Cantidad vendida Producto 3 (Q-P3)",  lambda v: validar_int(v,   "Q-P3",   1, 10000)),
        ("Cantidad vendida Producto 4 (Q-P4)",  lambda v: validar_int(v,   "Q-P4",   1, 10000)),
        ("Ventas S/ Producto 1 (S-P1)",         lambda v: validar_float(v, "S-P1",   0, 100000)),
        ("Ventas S/ Producto 2 (S-P2)",         lambda v: validar_float(v, "S-P2",   0, 100000)),
        ("Ventas S/ Producto 3 (S-P3)",         lambda v: validar_float(v, "S-P3",   0, 100000)),
        ("Ventas S/ Producto 4 (S-P4)",         lambda v: validar_float(v, "S-P4",   0, 100000)),
        ("Mes (1-12)",                           lambda v: validar_int(v,   "Mes",    1, 12)),
        ("Día de semana (0=Lun … 6=Dom)",        lambda v: validar_int(v,   "Día",    0, 6)),
    ]

    valores = []
    for etiqueta, validador in campos:
        while True:
            try:
                entrada = input(f"  → {etiqueta}: ").strip()
                valores.append(validador(entrada))
                break
            except ValueError as e:
                print(f"    [ERROR] {e}. Intente nuevamente.")

    print("\n  Seleccione algoritmo de predicción:")
    algs = list(resultados.keys())
    for i, a in enumerate(algs, 1):
        print(f"    {i}. {a}")
    while True:
        try:
            sel = int(input("  → Número: ")) - 1
            modelo_sel = algs[sel]
            break
        except:
            print("    [ERROR] Selección inválida.")

    resultado = predecir_venta_real(*valores, modelo_nombre=modelo_sel)
    print("\n" + "─" * 62)
    print("  RESULTADO DE LA PREDICCIÓN")
    print("─" * 62)
    for k, v in resultado.items():
        print(f"  {k.upper():<22}: {v}")
    print("─" * 62)

# =============================================================================
# 7. RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 62)
print("  RESUMEN DE RESULTADOS – Dataset statsfinal.csv")
print("=" * 62)
mejor = max(resultados, key=lambda m: resultados[m]['auc'])
print(f"{'Algoritmo':<25} {'Accuracy':>10} {'AUC-ROC':>10}")
print("-" * 48)
for nombre in resultados:
    marca = " ◄ MEJOR" if nombre == mejor else ""
    print(f"{nombre:<25} {resultados[nombre]['acc']:>10.4f} "
          f"{resultados[nombre]['auc']:>10.4f}{marca}")
print(f"\n  Registros usados : {len(df)}")
print(f"  Features         : {len(features)}")
print(f"  Split train/test : 80% / 20%")
print(f"  Modelo sugerido  : {mejor}")
print("=" * 62)
interfaz_consola()



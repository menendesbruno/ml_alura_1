import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, RFE, RFECV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random

SEED = 1234
random.seed(SEED)

df = pd.read_excel('../ML_Classificacao/resource/exames.xlsx')

valores_exames = df.drop(columns=['id', 'diagnostico', 'exame_33'])
diagnostico = df.diagnostico

teste_x, treino_x, teste_y, treino_y = train_test_split(valores_exames,
                                                        diagnostico,
                                                        test_size=0.3)

classificador = RandomForestClassifier(n_estimators=100)
classificador.fit(treino_x, treino_y)
score = classificador.score(teste_x, teste_y) * 100
print('Resultado da classificação: %.2f' % score)

classificador_bobo = DummyClassifier(strategy='most_frequent')
classificador_bobo.fit(treino_x, treino_y)
score_bobo = classificador_bobo.score(teste_x, teste_y) * 100
print('Resultado da classificação Boba: %.2f' % score_bobo)


pradonizador = StandardScaler()
pradonizador.fit(valores_exames)
valores_exames_v2 = pradonizador.transform(valores_exames)
valores_exames_v2 = pd.DataFrame(data=valores_exames_v2,
                                 columns=valores_exames.columns)

valores_exames_v3 = valores_exames_v2.drop(columns=['exame_4', 'exame_29'])


def grafico_violino(valores, inicio, fim):
    dados_plot = pd.concat([diagnostico, valores.iloc[:, inicio:fim]], axis=1)
    dados_plot = pd.melt(dados_plot, id_vars='diagnostico', var_name='exames',
                         value_name='valores')

    plt.figure(figsize=(10, 10))
    sns.violinplot(x='exames', y='valores', hue='diagnostico',
                   data=dados_plot, split=True)


grafico_violino(valores_exames_v2, 10, 21)
grafico_violino(valores_exames_v2, 22, 32)

matriz_correlacao = valores_exames_v3.corr()
plt.figure(figsize=(17, 15))
sns.heatmap(matriz_correlacao, annot=True, fmt='.1f')

matriz_correlacao_v1 = matriz_correlacao[matriz_correlacao > 1]
matriz_correlacao_v2 = matriz_correlacao_v1.sum()

select_kmelhores = SelectKBest(chi2, k=5)
valores_exames_v4 = valores_exames.drop(columns=['exame_4', 'exame_29', 'exame_3', 'exame_24'])
teste_x, treino_x, teste_y, treino_y = train_test_split(valores_exames_v4,
                                                        diagnostico,
                                                        test_size=0.3)

select_kmelhores.fit(treino_x, treino_y)
treino_kmelhores = select_kmelhores.transform(treino_x)
teste_kmelhores = select_kmelhores.transform(teste_x)

classificador = RandomForestClassifier(n_estimators=100)
classificador.fit(treino_kmelhores, treino_y)
score = classificador.score(teste_kmelhores, teste_y) * 100
print('Resultado da classificação kbest: %.2f' % score)
previsto = classificador.predict(teste_kmelhores)

matriz_confusao = confusion_matrix(teste_y, previsto)
print(matriz_confusao)
plt.figure(figsize=(17, 15))
sns.set()
sns.heatmap(matriz_confusao, annot=True, fmt='d').set(xlabel='Predicao', ylabel='Real')

teste_x, treino_x, teste_y, treino_y = train_test_split(valores_exames_v4,
                                                        diagnostico,
                                                        test_size=0.3)
estimator = RandomForestClassifier(n_estimators=100)
estimator.fit(treino_x, treino_y)
selecionar_rfe = RFE(estimator, n_features_to_select=5, step=2)
selecionar_rfe.fit(treino_x, treino_y)
treino_rfe = selecionar_rfe.transform(treino_x)
teste_rfe = selecionar_rfe.transform(teste_x)
classificador.fit(treino_rfe, treino_y)
score = classificador.score(teste_rfe, teste_y) * 100
print('Resultado da classificação RFE: %.2f' % score)

teste_x, treino_x, teste_y, treino_y = train_test_split(valores_exames_v4,
                                                        diagnostico,
                                                        test_size=0.3)
estimator = RandomForestClassifier(n_estimators=100)
estimator.fit(treino_x, treino_y)
selecionar_rfecv = RFECV(estimator, cv=5, step=1, scoring='accuracy')
selecionar_rfecv.fit(treino_x, treino_y)
treino_rfecv = selecionar_rfecv.transform(treino_x)
teste_rfecv = selecionar_rfecv.transform(teste_x)
estimator.fit(treino_rfecv, treino_y)
score = estimator.score(teste_rfecv, teste_y) * 100
print('Resultado da classificação RFECV: %.2f' % score)

# valores_exames_v5 = selecionar_rfe.transform(valores_exames_v4)
# print(valores_exames_v5.shape)
# sns.scatterplot(x=valores_exames_v5[:, 0], y=valores_exames_v5[:, 1], hue=diagnostico)

pca = PCA(n_components=2)
valores_exames_v5 = pca.fit_transform(valores_exames_v4)

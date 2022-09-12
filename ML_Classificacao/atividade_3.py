import pandas as pd
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('./resource/busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

X_dummies = pd.get_dummies(X_df)
Y_dummies = Y_df

X = X_dummies.values
Y = Y_dummies.values

percentual_treino = 0.8
percentual_teste = 0.1
tamanho_treino = int(percentual_treino * len(Y))
tamanho_teste = int(percentual_teste * len(Y))
tamanho_validacao = len(Y) - tamanho_treino - tamanho_teste


treino_dados = X[0:tamanho_treino]
treino_marcacoes = Y[0:tamanho_treino]
teste_dados = X[tamanho_treino:tamanho_treino + tamanho_teste]
teste_marcacoes = Y[tamanho_treino:tamanho_treino + tamanho_teste]
validacao_dados = X[-tamanho_treino + tamanho_teste:]
validacao_marcacoes = Y[tamanho_treino + tamanho_teste:]


def fit_and_predict(nome, modelo):
    model = modelo()
    model.fit(treino_dados, treino_marcacoes)
    resultado = model.predict(teste_dados)
    acertos = resultado == teste_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100.0 * (total_de_acertos / total_de_elementos)
    print(f"Taxa de acerto {nome}: ", taxa_de_acerto)
    return taxa_de_acerto


acerto_adaboost = fit_and_predict("AdaBoostClassifier", AdaBoostClassifier)
acerto_multinomial = fit_and_predict("MultinominalNB", MultinomialNB)

if acerto_multinomial > acerto_adaboost:
    vencedor = MultinomialNB()
else:
    vencedor = AdaBoostClassifier()


vencedor.fit(treino_dados, treino_marcacoes)

resultado = vencedor.predict(validacao_dados)
acertos = resultado == validacao_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_dados)

taxa_de_acerto = 100.0 * (total_de_acertos / total_de_elementos)
print(f"Taxa de acerto {vencedor} no mundo real: ", taxa_de_acerto)

acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100 * (acerto_base / len(teste_marcacoes))
print("Taxa de acerto base: ", taxa_de_acerto_base)
print(len(teste_marcacoes))


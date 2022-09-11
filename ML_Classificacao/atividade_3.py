import pandas as pd
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('./resource/busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

X_dummies = pd.get_dummies(X_df)
Y_dummies = Y_df

X = X_dummies.values
Y = Y_dummies.values

percentual_treino = 0.9
tamanho_treino = int(percentual_treino * len(Y))
tamanho_teste = len(Y) - tamanho_treino

treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]
teste_dados = X[-tamanho_teste:]
teste_marcacoes = Y[-tamanho_teste:]

model = MultinomialNB()
model.fit(treino_dados, treino_marcacoes)

resultado = model.predict(teste_dados)
acertos = resultado == teste_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * (total_de_acertos / total_de_elementos)

acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100 * (acerto_base / len(teste_marcacoes))

print("Taxa de acerto base: ", taxa_de_acerto_base)
print("Taxa de acerto algoritmo: ", taxa_de_acerto)
print(total_de_elementos)


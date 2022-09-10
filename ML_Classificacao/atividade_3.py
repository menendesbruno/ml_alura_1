import pandas as pd
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

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)


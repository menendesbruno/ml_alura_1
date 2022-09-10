from dados import carregar_acesso
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acesso()
treino_dados = X[:90]
treino_marcacoes = Y[:90]
teste_dados = X[-9:]
teste_marcacoes = Y[-9:]

model = MultinomialNB()
model.fit(treino_dados, treino_marcacoes)

resultado = model.predict(teste_dados)
diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
taxa_de_acerto = 100.0 * (len(acertos)/len(teste_dados))

print(round(taxa_de_acerto, 2))
print(len(teste_dados))

import pandas as pd
from dados import carregar_dados

df_acessos = pd.read_csv('./resource/acesso.csv')
df_cursos = pd.read_csv('./resource/cursos.csv')

X, Y = carregar_dados()

print(X)
print(Y)

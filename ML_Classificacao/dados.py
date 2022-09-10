import csv


def carregar_acesso():
    X = []
    Y = []
    arquivo = open('./resource/acesso.csv', 'r')
    leitor = csv.reader(arquivo)
    leitor.__next__()
    for home, como_funciona, contato, comprou in leitor:
        dados = [int(home), int(como_funciona), int(contato)]
        X.append(dados)
        Y.append(int(comprou))
    return X, Y


def carregar_busca():
    X = []
    Y = []
    arquivo = open('./resource/busca.csv', 'r')
    leitor = csv.reader(arquivo)
    leitor.__next__()
    for home, busca, logado, comprou in leitor:
        dados = [int(home), busca, int(logado)]
        X.append(dados)
        Y.append(int(comprou))
    return X, Y

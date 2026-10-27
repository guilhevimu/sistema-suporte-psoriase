import csv
import hashlib
import random
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

# ============================================================
# CONFIGURAÇÕES
# ============================================================
BASE_DIR = Path(
    r"C:\Users\rammo\OneDrive\Desktop\Mestrado\Modelagem"
)

CLASSES = {
    "PSOR": BASE_DIR / (
        "PSOR - GUILHERME - Tronco Menbros e Nadegas - 541 imagens"
    ),
    "NonPsor": BASE_DIR / (
        "NonPsor - Guilherme - Tronco Membros e Nadegas - 441 imagens"
    ),
}

SEMENTE = 42
PROPORCAO_TREINO = 0.70
PROPORCAO_VALIDACAO = 0.15
EXTENSOES = {".jpg", ".jpeg", ".png"}

# ============================================================
# IDENTIFICAR IMAGENS IDÊNTICAS PELOS PIXELS
# ============================================================
def assinatura_imagem(caminho):
    with Image.open(caminho) as imagem:
        imagem = imagem.convert("RGB")
        imagem.load()

        conteudo = (
            str(imagem.size).encode("utf-8")
            + imagem.tobytes()
        )

        return hashlib.sha256(conteudo).hexdigest()


# Também considera equivalência após aplicar orientação EXIF.
def assinatura_orientada(caminho):
    with Image.open(caminho) as imagem:
        imagem = ImageOps.exif_transpose(imagem).convert("RGB")
        imagem.load()

        conteudo = (
            str(imagem.size).encode("utf-8")
            + imagem.tobytes()
        )

        return hashlib.sha256(conteudo).hexdigest()


# ============================================================
# LER E VALIDAR AS ORIGENS ANTES DE MOVER OS CONJUNTOS
# ============================================================
encontradas = []
assinaturas_por_classe = {}
duplicatas = []
unicas_por_classe = {classe: [] for classe in CLASSES}

for classe, pasta in CLASSES.items():
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada:\n{pasta}")

    arquivos = sorted(
        (
            arquivo
            for arquivo in pasta.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in EXTENSOES
        ),
        key=lambda arquivo: arquivo.name.lower(),
    )

    if not arquivos:
        raise ValueError(f"Nenhuma imagem encontrada em:\n{pasta}")

    for arquivo in arquivos:
        try:
            hashes = {
                assinatura_imagem(arquivo),
                assinatura_orientada(arquivo),
            }
        except Exception as erro:
            raise RuntimeError(
                f"Não foi possível ler a imagem:\n{arquivo}"
            ) from erro

        encontradas.append({
            "classe": classe,
            "arquivo": arquivo,
            "hashes": hashes,
        })

# ============================================================
# AGRUPAR IMAGENS EQUIVALENTES
# Uma imagem pode conectar grupos por suas duas assinaturas.
# ============================================================
pais = list(range(len(encontradas)))


def raiz(indice):
    while pais[indice] != indice:
        pais[indice] = pais[pais[indice]]
        indice = pais[indice]
    return indice


def unir(a, b):
    a, b = raiz(a), raiz(b)
    if a != b:
        pais[b] = a


indice_por_hash = {}

for indice, registro in enumerate(encontradas):
    for assinatura in registro["hashes"]:
        if assinatura in indice_por_hash:
            unir(indice, indice_por_hash[assinatura])
        else:
            indice_por_hash[assinatura] = indice

grupos = {}

for indice, registro in enumerate(encontradas):
    grupos.setdefault(raiz(indice), []).append(registro)

conflitos = []

for grupo in grupos.values():
    classes_grupo = {registro["classe"] for registro in grupo}

    if len(classes_grupo) > 1:
        conflitos.append(grupo)
        continue

    classe = grupo[0]["classe"]
    principal = grupo[0]["arquivo"]
    unicas_por_classe[classe].append(principal)

    for registro in grupo[1:]:
        duplicatas.append({
            "Classe": classe,
            "Imagem_mantida": str(principal),
            "Copia_nao_distribuida": str(registro["arquivo"]),
        })

if conflitos:
    print("\nIMAGENS IDÊNTICAS COM RÓTULOS DIFERENTES:")

    for numero, grupo in enumerate(conflitos, start=1):
        print(f"\nConflito {numero}:")
        for registro in grupo:
            print(f"  {registro['classe']}: {registro['arquivo']}")

    raise ValueError(
        "Confira os rótulos nas pastas de origem e execute novamente. "
        "Os conjuntos atuais não foram modificados."
    )

# ============================================================
# PLANEJAR DIVISÃO APENAS COM IMAGENS ÚNICAS
# ============================================================
divisoes = {}

for classe, arquivos in unicas_por_classe.items():
    imagens = sorted(arquivos, key=lambda arquivo: arquivo.name.lower())
    random.Random(SEMENTE).shuffle(imagens)

    total = len(imagens)
    n_treino = int(total * PROPORCAO_TREINO)
    n_validacao = int(total * PROPORCAO_VALIDACAO)
    n_teste = total - n_treino - n_validacao

    if min(n_treino, n_validacao, n_teste) == 0:
        raise ValueError(
            f"Imagens únicas insuficientes para dividir a classe {classe}."
        )

    divisoes[classe] = {
        "Treino": imagens[:n_treino],
        "Validacao": imagens[n_treino:n_treino + n_validacao],
        "Teste": imagens[n_treino + n_validacao:],
    }

# ============================================================
# PRESERVAR CONJUNTOS ANTERIORES
# ============================================================
identificador = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
backup = BASE_DIR / "Backup_Separacoes" / identificador

for conjunto in ("Treino", "Validacao", "Teste"):
    pasta_atual = BASE_DIR / conjunto

    if pasta_atual.exists():
        backup.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pasta_atual), str(backup / conjunto))

# ============================================================
# COPIAR E REGISTRAR A NOVA DISTRIBUIÇÃO
# ============================================================
manifesto = []

for classe, conjuntos in divisoes.items():
    for conjunto, imagens in conjuntos.items():
        destino = BASE_DIR / conjunto / classe
        destino.mkdir(parents=True, exist_ok=True)

        for origem in imagens:
            arquivo_destino = destino / origem.name
            shutil.copy2(origem, arquivo_destino)

            manifesto.append({
                "Classe": classe,
                "Conjunto": conjunto,
                "Origem": str(origem),
                "Destino": str(arquivo_destino),
                "Semente": SEMENTE,
            })


def salvar_csv(caminho, registros):
    if not registros:
        return

    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=list(registros[0]),
            delimiter=";",
        )
        escritor.writeheader()
        escritor.writerows(registros)


salvar_csv(
    BASE_DIR / f"distribuicao_imagens_{identificador}.csv",
    manifesto,
)

salvar_csv(
    BASE_DIR / f"copias_nao_distribuidas_{identificador}.csv",
    duplicatas,
)

# ============================================================
# RESUMO
# ============================================================
print("\nDistribuição de imagens únicas:")
print(
    f"{'Classe':<12}"
    f"{'Treino':>10}"
    f"{'Validação':>12}"
    f"{'Teste':>10}"
    f"{'Total':>10}"
)
print("-" * 54)

for classe, conjuntos in divisoes.items():
    print(
        f"{classe:<12}"
        f"{len(conjuntos['Treino']):>10}"
        f"{len(conjuntos['Validacao']):>12}"
        f"{len(conjuntos['Teste']):>10}"
        f"{len(unicas_por_classe[classe]):>10}"
    )

print(f"\nArquivos encontrados nas origens: {len(encontradas)}")
print(f"Cópias redundantes não distribuídas: {len(duplicatas)}")
print(f"Imagens distribuídas: {len(manifesto)}")
print("Arquivos originais preservados.")

if backup.exists():
    print(f"Backup dos conjuntos anteriores:\n{backup}")

print("\nSeparação concluída.")
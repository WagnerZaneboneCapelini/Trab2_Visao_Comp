# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Wagner e João Victor Alves

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv
import random

########################################################################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)


def normalize_points(points):
    # Calculate centroid
    centroid = np.mean(points, axis=0)  # Retorna [cx, cy]

    # Subtrai o centroide de cada ponto e calcula a norma (distância Euclidiana)
    dists = np.linalg.norm(points - centroid, axis=1)
    mean_dist = np.mean(dists)

    scale = np.sqrt(2) / mean_dist

    # Esta matriz primeiro translada os pontos para a origem e depois os escala
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    # Adiciona uma coluna de '1s' e transpõe para o formato (3, N)
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))]).T

    # Normalizar os pontos
    norm_points = T @ homogeneous_points

    return norm_points, T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)


def my_DLT(pts1, pts2):
    # pts1 e pts2 já estão no formato homogêneo (3, N)
    A = []

    # Itera sobre cada par de pontos correspondentes
    for i in range(pts1.shape[1]):
        x1, y1, w1 = pts1[:, i]
        x2, y2, w2 = pts2[:, i]

        A.append([0, 0, 0, -w2*x1, -w2*y1, -w2*w1, y2*x1, y2*y1, y2*w1])
        A.append([w2*x1, w2*y1, w2*w1, 0, 0, 0, -x2*x1, -x2*y1, -x2*w1])

    A = np.array(A)

    # Tratamento de erro para a decomposição SVD
    try:
        U, S, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        # Se o SVD não convergir, retorna None para indicar falha
        return None

    # A solução para H é a última linha de V (ou a última coluna de V transposto)
    H_matrix = Vt[-1, :].reshape(3, 3)

    return H_matrix


'''
def compute_A(pts1, pts2):

    return A
'''
# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)


def compute_normalized_dlt(pts1, pts2):
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)

    # Estima a homografia H_normalizada
    H_norm = my_DLT(norm_pts1, norm_pts2)

    # Se my_DLT falhar, propague o erro
    if H_norm is None:
        return None

    # Denormaliza H_normalizada e obtém H
    H = np.linalg.inv(T2) @ H_norm @ T1

    return H


# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens


def RANSAC(pts1, pts2, dis_threshold, N, Ninl):
    """
    Executa o algoritmo RANSAC para encontrar uma homografia robusta.
    """
    # --- Configuração e Mapeamento de Parâmetros ---

    s = 6
    s_para_calcular_N = 4

    total_de_pontos = len(pts1)
    confianca_desejada = Ninl
    iteracoes_maximas = N

    melhor_conjunto_de_inliers = []

    # O loop principal pode ser executado menos vezes que N, devido à otimização
    for _ in range(iteracoes_maximas):
        # Amostra aleatória de 's' pontos
        indices_da_amostra = random.sample(range(total_de_pontos), s)
        amostra_pts1 = pts1[indices_da_amostra]
        amostra_pts2 = pts2[indices_da_amostra]

        # Cálculo do modelo (Homografia) com a amostra
        homografia_teste = compute_normalized_dlt(
            amostra_pts1.reshape(-1, 2),
            amostra_pts2.reshape(-1, 2)
        )

        # Se o modelo não pôde ser calculado, pula para a próxima iteração
        if homografia_teste is None:
            continue

        # Verificação de quais pontos se encaixam no modelo (consenso)
        inliers_desta_iteracao = []
        for i in range(total_de_pontos):
            ponto1_homogeneo = np.array([*pts1[i].flatten(), 1])
            ponto1_projetado_h = homografia_teste @ ponto1_homogeneo

            # Evita divisão por zero ao normalizar
            if ponto1_projetado_h[-1] == 0:
                continue

            ponto1_projetado = ponto1_projetado_h[:2] / ponto1_projetado_h[-1]
            ponto2_real = pts2[i].flatten()

            # Calcula a distância euclidiana para verificar se é um inlier
            distancia_reprojecao = np.linalg.norm(
                ponto1_projetado - ponto2_real)
            if distancia_reprojecao <= dis_threshold:
                inliers_desta_iteracao.append(i)

        # Avaliação e atualização do melhor modelo encontrado
        if len(inliers_desta_iteracao) > len(melhor_conjunto_de_inliers):
            melhor_conjunto_de_inliers = inliers_desta_iteracao

            # Otimização: Recalcula o número de iterações necessárias (N)
            proporcao_de_inliers = len(
                melhor_conjunto_de_inliers) / total_de_pontos
            numerador = np.log(1 - confianca_desejada)
            denominador = np.log(1 - proporcao_de_inliers ** s_para_calcular_N)

            if denominador != 0:
                # Atualiza o número de iterações do loop, potencialmente terminando mais cedo
                iteracoes_maximas = int(numerador / denominador)

            # Otimização: Critério de parada antecipada
            if len(melhor_conjunto_de_inliers) > total_de_pontos * confianca_desejada:
                print('Quantidade de pontos suficiente atendida (parada antecipada).')
                break

    # Finalização
    if not melhor_conjunto_de_inliers:
        print('Não foi possível encontrar um modelo consistente.')
        return None

    if _ == N - 1:
        print('Número máximo de iterações atendido.')

    print(
        f'Modelo final será calculado com o melhor conjunto de {len(melhor_conjunto_de_inliers)} inliers.')

    # Recalcula a homografia final com todos os inliers do melhor conjunto
    inliers_pts1 = pts1[melhor_conjunto_de_inliers]
    inliers_pts2 = pts2[melhor_conjunto_de_inliers]

    H_ransac = compute_normalized_dlt(
        inliers_pts1.reshape(-1, 2),
        inliers_pts2.reshape(-1, 2)
    )

    return H_ransac
########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
# -------- TESTE 1 ------------------------------
# img2 = cv.imread('mesa02_1.jpg', 0)  # queryImage
# img1 = cv.imread('mesa_livro01.jpg', 0)  # trainImage

# --------- TESTE 2 -----------------------------
img1 = cv.imread('livro_001.jpg', 0)  # queryImage
img2 = cv.imread('livro_002.jpg', 0)  # trainImage

# --------- TESTE 3 -----------------------------
# img1 = cv.imread('comicsStarWars01.jpg', 0)  # queryImage
# img2 = cv.imread('comicsStarWars02.jpg', 0)  # trainImage


# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # O N=100 e Ninl=0.96 já atende aos 3 testes que eu fiz, mas por garantia coloquei N=300
    sigma = 1
    dis_threshold = 6*(sigma**2)
    M = RANSAC(src_pts, dst_pts, dis_threshold, N=300, Ninl=0.96)

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################

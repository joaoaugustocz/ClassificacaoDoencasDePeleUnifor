
import cv2
import numpy as np

def segment_lesion(image_path):
    """
    Esta função segmenta e isola uma lesão de pele em uma imagem.

    Args:
        image_path (str): O caminho para a imagem a ser processada.
    """
    # 1. Carregar a imagem
    # Carregamos a imagem do caminho fornecido.
    img = cv2.imread(image_path)

    # 2. Pré-processamento
    # Convertemos a imagem para o espaço de cores LAB. O canal 'a' (cores do verde ao magenta)
    # é particularmente útil para destacar lesões de pele.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:,:,1]

    # Aplicamos um desfoque gaussiano para suavizar a imagem e reduzir o ruído.
    blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)

    # 3. Segmentação
    # Usamos o método de limiarização de Otsu para encontrar um valor de limiar ideal
    # e criar uma máscara binária. A limiarização de Otsu assume que a imagem
    # contém dois tipos de pixels (fundo e objeto) e encontra o limiar que
    # minimiza a variância intra-classe.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Pós-processamento
    # Usamos operações morfológicas para limpar a máscara.
    # A "abertura" (opening) remove pequenos ruídos brancos no fundo.
    # O "fechamento" (closing) preenche pequenos buracos pretos na lesão.
    #kernel = np.ones((3,3),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 3)

    # 5. Encontrar e desenhar contornos
    # Encontramos os contornos na máscara. Um contorno é uma curva que une
    # todos os pontos contínuos ao longo do limite que têm a mesma cor ou intensidade.
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criamos uma máscara preta para desenhar o contorno preenchido.
    mask = np.zeros(img.shape[:2], dtype="uint8")

    # Assumimos que o maior contorno corresponde à lesão e o desenhamos na máscara.
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, (255), thickness=cv2.FILLED)

    # 6. Isolar a lesão
    # Usamos a máscara para isolar a lesão da imagem original.
    # A operação bitwise_and copia os pixels da imagem original apenas onde a máscara é branca.
    isolated_lesion = cv2.bitwise_and(img, img, mask=mask)

    # 7. Salvar os resultados
    # Salvamos a máscara e a lesão isolada para visualização.
    cv2.imwrite("lesion_mask.jpg", mask)
    cv2.imwrite("isolated_lesion.jpg", isolated_lesion)

    print("Segmentação concluída. A máscara da lesão e a lesão isolada foram salvas.")

# Executar a função com a imagem de teste
segment_lesion(r"C:\Users\czjoa\Documents\AulasUnifor\PDI\ISIC_0025018.jpg")

import cv2
import numpy as np

"""

    num_components = num de eigenvectors por id(pessoa)
    threshold: distância euclidiana mínima que delimita se uma imagem de teste pertence/não pertence a base de dados
"""

detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml") #para utilizar o reconhecedor é necessário passar como parâmetro o classificador criado
largura, altura = 220, 220 #220 x 220 pixels = 48400 pixels no total
font = cv2.FONT_HERSHEY_COMPLEX_SMALL #font usada para imprimir o nome da pessoa junto com a face
camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #converte a imagem capturada pela webcam para tons de cinza

    #chamamos o detector de faces, passando como parâmetro a imagem em tons de cinza. minSize (30x30) é a largura e altura mínima que pode ser detectada pelo detector de faces
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.5,
                                                    minSize=(30, 30))
    # percorre as faces detectadas
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura)) # converte a imagem caputrada em tons de cinza para o tamanho de 220x220
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255), 2) # desenha o retângulo em volta da face. (0, 0, 255) é em BG(R). 2 é a espessura da borda
        id, confianca = reconhecedor.predict(imagemFace) # o reconhecedor faz a predição de qual classe a imagem detectada pela câmera pertence e diz o grau de confiança da predicao/classificacao. confianca é a menor distancia euclidiana (threshold) da imagem de teste para cada uma das eigenfaces da base de dados
        nome = ""
        if id == 1:
            nome = 'Bruno'
        elif id == 2:
            # se colocasse só o else eu estaria desconsiderando o threshold. Quando eu seto o threshold, o id retornado pode ser da base de dados ou vazio(NONE ou algo do tipo), indicando que a imagem de teste nao pertence a base de dados
            nome = 'Mae'
        elif id == 3:
            nome = 'Pai'

        cv2.putText(imagem, nome, (x,y +(a+30)), font, 2, (0,0,255))
        cv2.putText(imagem, str(confianca), (x,y + (a+50)), font, 1, (0,0,255)) # coloca um texto com o id da face reconhecida com a fonte, o tamanho da fonte e a cor vermelha. a+50  é posicao. font é a fonte. 1 é o tamanho. (B=0, G=0, R=255) é a cor

    cv2.imshow(winname="Face", mat=imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
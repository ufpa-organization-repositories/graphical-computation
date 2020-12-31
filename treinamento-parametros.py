import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
# eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold=2)

fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

#percorre as imagens de treinamento e retorna uma lista com os ids e as imagens dos ids
def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    # print(caminhos)
    faces = []
    ids = []

    for caminhoImagem in caminhos:

        # lê cada uma das imagens do diretório
        # imagemFace = cv2.imread(caminhoImagem)

        # converte para escala de cinza para o treinamento dos classificadores
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)

        # cv2.imshow(winname="Face", mat=imagemFace)
        # cv2.waitKey(10) #10 milisegundos para processar cada uma das imagens

        # pega o id de cada foto através do path da foto
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])

        # id_int = int(id_string_ordinal)
        ids.append(id)
        faces.append(imagemFace)

        #np.array é o tipo do parametro requerido para fazer o treinamento

    return np.array(ids), faces

ids, faces = getImagemComId()

# for id, face in zip(ids, faces):
#     #note que a posicao na lista do id e da face remetem ao mesmo arquivo
#     print(id)
#     print(face)

print("Treinando os classificadores")

# aqui ocorre um aprendizado supervisionado, pois são passadas as imagens e seus respectivos ids
# n_pessoas > 1 para poder treinar

eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml') #esse arquivo que é o classificador que será utilizado para fazer o reconhecimento facial

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado")
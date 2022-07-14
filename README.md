# SkinLesionClassifier

Construcao de um aplicativo no Streamlit para a classificacao de lesoes de pele.

Link para o aplicativo:
https://luccafurtado-classifier-stlit-4yklz4.streamlitapp.com/

O usuario pode fazer o upload de uma imagem de lesao de pele e submeter para classificacao feita por um modelo.
Este fara a predicao e retornara a classe de maior probabilidade, realizando a classificacao da lesao.

Para teste do aplicativo, temos as seguintes imagens:
#### OBS: As imagens a seguir são fotos de lesões de pele. Algumas pessoas se sentem desconfortáveis com esse tipo de imagem.

[Melanoma](https://github.com/LuccaFurtado/images/blob/main/melanoma.jpeg)

[Vascular lesion](https://github.com/LuccaFurtado/images/blob/main/vascular_lesion.jpeg)

[Actinic Keratosis](https://github.com/LuccaFurtado/images/blob/main/akiec.jpg)

[Basal Cell Carcinoma](https://github.com/LuccaFurtado/images/blob/main/bcc.jpg)

[Melanocytic Nevus](https://github.com/LuccaFurtado/images/blob/main/nv.jpg)


O classificador, um modelo de rede neural convolucional, foi treinado utilizando dezenas de milhares de imagens de lesao de pele da base de dados do ISIC(The International Skin Imaging Collaboration).

O processo de treinamento pode ser visto em: https://github.com/LuccaFurtado/Classifier/blob/main/train_skin_lesion.ipynb



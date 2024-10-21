# Sobre o Projeto

Este é projeto de um pacote Python capaz de treinar uma *Fully Covolutional Network* (FCN) para segmentação semântica de imagens.

# Obtenção de Imagens

Sites como o [Kaggle](https://www.kaggle.com/) e [images.cv](https://images.cv/) podem ser usados para obter a base de dados das imagens. Note que pode ser interessante remover ou filtrar algumas imagens obtidas destes sites por não serem ideais para certas aplicações de ML.

# Configuração de Parâmetros

É possível configurar os parâmetros de pré-processamento e treino pelo arquivo `params.py`. Segue abaixo um exemplo desse arquivo.

```python
color_mode = 'RGB'
img_size = 128

epochs = 30
```

* `color_mode`: Modo de cor das imagens pré-processadas. Pode ser 'L' para escala de cinza e 'RGB' para a escala RGB
* `img_size`: Valor para o qual as imagens serão redimensionadas
* `epochs`: Quantidade de épocas do treinamento do zero usando Tensorflow

As imagens de treinamento devem ser salvas na pasta `imgs`. Por padrão, a pasta `imgs` contem imagens de algumas beringelas

A pasta `test_imgs` deve conter imagens definidas pelo usuário que serão usadas para testar o modelo pronto. Por padrão, esta pasta contém algumas imagens de beringelas pegas aleatoriamente na internet.

# Executando o Código

## Instalando Depedências

`poetry install`

## Realizando Pré-processamento

`poetry run preprocess`

Este comando irá gerar a pasta `tmp` contendo as imagens pré-processadas.

## Marcando Imagens

```
labelme tmp/processed_imgs --output tmp/labels/
```

Este comando deve abrir uma GUI que permitirá marcar os *targets* das imagens em `tmp/processed_imgs`. Para cada imagem, um arquivo  `.json` será criado em `tmp/labels`.

## Realizando Treinamento

`poetry run train`

O modelo treinado será exportado para `tmp/model`.

## Testando Modelo

`poetry run test`

Este comando irá segmentar as imagens na pasta `test_imgs`, salvando os resultados em `tmp/test_results`.

# Usa uma imagem base oficial do Python
FROM python:3.9-slim

# Maintainer
LABEL maintainer="Brenda Verch <brenda_verch@hotmail.com>"

# Cria uma pasta /app no container
WORKDIR /app

# Copia o arquivo de requisitos primeiro
COPY requirements.txt /app

# Instala as bibliotecas
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o restante do código
COPY . /app

# Roda o script que treina o modelo (gera modelo_rf.pkl e input_columns.pkl)
RUN python model/train_model.py

# Expõe a porta usada pela API Flask
EXPOSE 5000

# Comando padrão ao iniciar o container
CMD ["python", "app/main.py"]

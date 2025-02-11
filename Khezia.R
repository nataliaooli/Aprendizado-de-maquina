########### Trabalho Final - Tópicos 2 Profª Thais ##############

# carregando pacotes

pacman::p_load(tidyverse, tidymodels, rsample)
tidymodels_prefer()

# leitura dos dados

# dados extraídos do Kaggle no dia 10/02/2025
dados <- read.csv(file = "World-happiness-report-updated_2024.csv", 
                  sep = ",")

dim(dados)

# divisão dos dados em treino/validação/teste
set.seed(2025)
dados_split <- initial_validation_split(ames, prop = c(0.6, 0.2))
dados_split

dados_treino <- training(dados_split)
dados_val <- validation(dados_split)
dados_teste <- testing(dados_split)

# Ajustar dois algoritmos de regressão de aprendizado de máquina. Algumas opções são:
# Árvore de decisão
# Floresta Aleatória
# Regressão Ridge
# Regressão Lasso
# Support Vector Regression (versão do SVM para regressão)
# Regressão Linear
# Regressão Polinomial
# GMM ???


########### Trabalho Final - Tópicos 2 Profª Thais ##############

# carregando pacotes ####

pacman::p_load(tidyverse, tidymodels, rsample, ranger, kernlab)
tidymodels_prefer()

# leitura dos dados ####

# dados extraídos do Kaggle no dia 10/02/2025 
dados <- read.csv(file = "World-happiness-report-updated_2024.csv", 
                  sep = ",") %>% 
  select("Life.Ladder", "Social.support", "Healthy.life.expectancy.at.birth",
           "Freedom.to.make.life.choices", "Perceptions.of.corruption", "Generosity")

dim(dados)


# com base na análise PCA que a Khezia fez, vamos utilizar as variáveis: 
# variável resposta: Life.Ladder
# variáveis explicativas:
# - suporte social (Social.support)
# - expectativa de vida (Healthy.life.expectancy.at.birth)
# - Liberdade para escolhas (Freedom.to.make.life.choices)
# - Corrupção (Perceptions.of.corruption)
# - Generosidade (Generosity)
names(dados)
str(dados)

min(dados$Life.Ladder)

# divisão dos dados em treino/validação/teste ####
set.seed(2025)
dados_split <- initial_validation_split(dados, prop = c(0.6, 0.2))
# separar para que tenha amostras de cada país em cada conjunto não deu por conta do tamanho da base.
# a separação foi sem os estratos

dados_split

dados_treino <- training(dados_split)
dados_val <- validation(dados_split)
dados_teste <- testing(dados_split)

# Recipe ####
dados_treino_recipe <- 
  recipe(Life.Ladder ~ Social.support + Healthy.life.expectancy.at.birth +
           Freedom.to.make.life.choices + Perceptions.of.corruption + Generosity,
         data = dados_treino) %>%
  step_impute_mean(Social.support, Healthy.life.expectancy.at.birth, Freedom.to.make.life.choices,
                   Perceptions.of.corruption, Generosity)


#Modelo SVM ####
svm_reg <- 
  svm_rbf(cost = 1, margin =0.005) %>% 
  set_mode("regression") %>% 
  set_engine("kernlab")

svm_reg_wflow <- 
  workflow() %>% 
  add_model(svm_reg) %>% 
  add_recipe(dados_treino_recipe)

# ajuste do modelo
svm_fit <- fit(svm_reg_wflow, dados_treino)

# conjunto de validação
svm_val_pred <- predict(svm_fit, dados_val)
validacao <- bind_cols(dados_val, svm_val_pred)

validacao %>% metrics(truth = Life.Ladder, estimate = .pred)




table(teste.1$Obesidade, pred1)

# svm_models <- workflow_set(list(regressao = dados_receita), list(svm = svm_reg), cross = TRUE)
# svm_models
# 
# 
# svm_models <- 
#   svm_models %>% 
#   workflow_map("fit_resamples", 
#                # Options to `workflow_map()`: 
#                seed = 2025, verbose = TRUE,
#                # Options to `fit_resamples()`: 
#                resamples = dados_folds, control = keep_pred)

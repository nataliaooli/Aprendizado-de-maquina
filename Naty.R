########### Trabalho Final - Tópicos 2 Profª Thais ##############

# carregando pacotes ####

pacman::p_load(tidyverse, tidymodels, rsample, ranger, kernlab, readxl, glmnet)
tidymodels_prefer()

# leitura dos dados ####

# dados extraídos do Kaggle no dia 10/02/2025 
dados <-read_xls(path = "world-happiness-report-2024-final.xls", range = "A1:K144") %>% 
  select("Ladder score", "Explained by: Social support", "Explained by: Freedom to make life choices",
         "Explained by: Healthy life expectancy", "Explained by: Generosity", 
         "Explained by: Perceptions of corruption")
nomes_colunas <- c("Ladder.score", "Social.support", "Freedom.to.make.life.changes",
                   "Healthy.life.expectancy", "Generosity", "Perception.of.corruption")

colnames(dados) <- nomes_colunas
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

# divisão dos dados em treino/validação/teste ####
set.seed(2025)
dados_split <- initial_validation_split(dados, prop = c(0.6, 0.2))
# separar para que tenha amostras de cada país em cada conjunto não deu por conta do tamanho da base.
# a separação foi sem os estratos

dados_split

dados_treino <- training(dados_split) #treino
dados_val <- validation(dados_split) # validação
dados_teste <- testing(dados_split) # teste

# Recipe ####
dados_treino_recipe <- 
  recipe(Ladder.score ~ Social.support + Healthy.life.expectancy +
           Freedom.to.make.life.changes + Perception.of.corruption + Generosity,
         data = dados_treino) %>%
  step_impute_mean(Social.support, Healthy.life.expectancy, Freedom.to.make.life.changes,
                   Perception.of.corruption, Generosity)

# Regressão Linear ####
reg_linear <- 
  linear_reg() %>% 
  set_engine("lm")

reg_linear_wflow <- 
  workflow() %>% 
  add_model(reg_linear) %>% 
  add_recipe(dados_treino_recipe)

# ajuste do modelo nos dados de treino
rl_fit <- fit(reg_linear_wflow, dados_treino)

# conjunto de validação
rl_val_pred <- predict(rl_fit, dados_val %>% select(-Ladder.score))
validacao <- bind_cols(dados_val, svm_val_pred)

validacao %>% metrics(truth = Ladder.score, estimate = .pred)

# Medidas de desempenho ####

metricas <- metric_set(rmse, rsq, mae)
metricas(predito_teste, truth = Ladder.score, estimate = .pred)

# Modelo ridge ####
# Criar um novo modelo de regressão linear com penalização (ridge)
modelo_ridge <- 
  linear_reg(penalty = 0.1, mixture = 1) %>%  # Penalty controla a regularização, mixture=0 indica Ridge
  set_engine("glmnet")

# Criar workflow
ridge_wflow <- 
  workflow() %>%
  add_model(modelo_ridge)%>% 
  add_recipe(dados_treino_recipe)

# Treinar o novo modelo no conjunto de treino
ridge_fit <- fit(ridge_wflow, data = dados_treino)

# Avaliar no conjunto de validação novamente
ridge_validation <- predict(ridge_fit, new_data = dados_val %>% select(-Ladder.score)) %>%
  bind_cols(dados_val %>% select(Ladder.score))

ridge_validation %>%
  metrics(truth = Ladder.score, estimate = .pred)

print(valid_metrics_ridge)



#Modelo SVM ####
svm_reg <- 
  svm_rbf(cost = 1, margin = 0.001)%>% 
  set_mode("regression") %>% 
  set_engine("kernlab")

svm_reg_wflow <- 
  workflow() %>% 
  add_model(svm_reg) %>% 
  add_recipe(dados_treino_recipe)

# ajuste do modelo nos dados de treino
svm_fit <- fit(svm_reg_wflow, dados_treino)

# conjunto de validação
svm_val_pred <- predict(svm_fit, dados_val %>% select(-Ladder.score))
validacao <- bind_cols(dados_val, svm_val_pred)

validacao %>% metrics(truth = Ladder.score, estimate = .pred)

# ajuste do modelo
predito_teste <- predict(svm_fit, dados_teste %>% select(-Ladder.score))
predito_teste <- bind_cols(predito_teste, dados_teste %>% select(Ladder.score))

ggplot(predito_teste, aes(x = Ladder.score, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2, color="red") + 
  geom_point(alpha = 0.5, size = 2, color = "blue") + 
  labs(y = "Predito - Índice de Felicidade", x = "Índice de Felicidade") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred()+ 
  theme_minimal()


# Medidas de desempenho ####

metricas <- metric_set(rmse, rsq, mae)
metricas(predito_teste, truth = Ladder.score, estimate = .pred)


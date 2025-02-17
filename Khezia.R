---
title: "Trabalho Machine Learning"
author: "Khézia R. de Moura | 180104322"
date: "2025-02-15"
output: html_document
---

```{r setup, include=FALSE}
pacman::p_load(tidyverse, tidymodels, rsample, readxl, ggcorrplot)
tidymodels_prefer()
```

```{r}
dados <- read_xls("world-happiness-report-2024.xls")

dados <- dados %>% rename("Suporte Social" = "Explained by: Social support",
                  "Expectativa de vida" = "Explained by: Healthy life expectancy",
                  "Liberdade para fazer escolhas" = "Explained by: Freedom to make life choices",
                  "Percepção de Corrupção" = "Explained by: Perceptions of corruption",
                  "Generosidade" = "Explained by: Generosity") 

dados <- dados %>% select(`Country name`, `Ladder score`, `Suporte Social`, `Expectativa de vida`, `Liberdade para fazer escolhas`, `Percepção de Corrupção`, `Generosidade`)

glimpse(dados) 

```

```{r}

summary(dados$`Ladder score`) %>% print()
var(dados$`Ladder score`   ) #1.27 

#boxplot
ggplot(dados, aes(x = "", y = `Ladder score`   )) +
  geom_boxplot(fill = "lightblue", color = "black") +
  theme_minimal() +
  labs(y = "Índice de Felicidade")

#histograma
ggplot(dados, aes(x = `Ladder score`   )) +
  geom_histogram(bins = 30, fill = "blue", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(x = "Índice de Felicidade", y = "Frequência")


# Transformando os dados para o formato longo
dados_long <- pivot_longer(dados, 
                           cols = c(`Suporte Social`, 
                                    `Expectativa de vida`,
                                    `Liberdade para fazer escolhas`,
                                    `Percepção de Corrupção`,
                                    `Generosidade`), 
                           names_to = "Variavel", 
                           values_to = "Valor")

# Criando o gráfico de histogramas
ggplot(dados_long, aes(x = Valor)) +
  geom_histogram(bins = 20, fill = "blue", color = "black", alpha = 0.7) +
  facet_wrap(~ Variavel, scales = "free") +  # Organiza os histogramas
  theme_minimal() +
  labs(x = "Valores", y = "Nota")


corr <- cor(dados[, -1], use = "complete.obs")
ggcorrplot(corr, method = "circle", type = "lower", lab = TRUE)

```

```{r}
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
  recipe(`Ladder score` ~ `Suporte Social` + `Expectativa de vida` +
           `Liberdade para fazer escolhas` + `Percepção de Corrupção` + Generosidade,
         data = dados_treino) %>%
  step_impute_mean(`Suporte Social`, `Expectativa de vida`, `Liberdade para fazer escolhas`,
                   `Percepção de Corrupção`, Generosidade)


# Definir o modelo de árvore de decisão para regressão
modelo_arvore <- 
  decision_tree(mode = "regression") %>% 
  set_engine("rpart")  # Usa o motor "rpart" para árvore de decisão

# Criar o workflow combinando o modelo e o recipe
workflow_arvore <- 
  workflow() %>%
  add_recipe(dados_treino_recipe) %>%
  add_model(modelo_arvore)

# Treinar o modelo com os dados de treino
modelo_treinado <- workflow_arvore %>%
  fit(data = dados_treino)

# Fazer previsões no conjunto de validação
previsoes_val <- modelo_treinado %>%
  predict(new_data = dados_val) %>%
  bind_cols(dados_val)

# Avaliar o modelo com métricas de desempenho
metricas <- metric_set(rmse, rsq)

resultados_val <- metricas(previsoes_val, truth = `Ladder score`, estimate = .pred)
print(resultados_val)

# Fazer previsões no conjunto de teste
previsoes_teste <- modelo_treinado %>%
  predict(new_data = dados_teste) %>%
  bind_cols(dados_teste)

# Avaliar no conjunto de teste
resultados_teste <- metricas(previsoes_teste, truth = `Ladder score`, estimate = .pred)
print(resultados_teste)


library(rpart.plot) 
# Extrair o modelo treinado da árvore de decisão
modelo_final <- extract_fit_parsnip(modelo_treinado)$fit

# Plotar a árvore de decisão
rpart.plot(modelo_final, type = 3, fallen.leaves = TRUE, cex = 0.8)

```
#Validaçao cruzada
```{r}
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
  recipe(`Ladder score` ~ `Suporte Social` + `Expectativa de vida` +
           `Liberdade para fazer escolhas` + `Percepção de Corrupção` + Generosidade,
         data = dados_treino) %>%
  step_impute_mean(`Suporte Social`, `Expectativa de vida`, `Liberdade para fazer escolhas`,
                   `Percepção de Corrupção`, Generosidade)


# Definir o modelo de árvore de decisão para regressão
modelo_arvore <- 
  decision_tree(mode = "regression") %>% 
  set_engine("rpart")  # Usa o motor "rpart" para árvore de decisão

# Criar resamples para validação cruzada (10 folds)
cv_folds <- vfold_cv(dados_treino, v = 10)

# Criar workflow
workflow_arvore <- 
  workflow() %>%
  add_recipe(dados_treino_recipe) %>%
  add_model(modelo_arvore)

# Ajustar a validação cruzada
cv_results <- workflow_arvore %>%
  fit_resamples(resamples = cv_folds, metrics = metric_set(rmse, rsq))

# Ver resultados médios das métricas
collect_metrics(cv_results)

```



# Chamar os pacotes necessários
library(randomForest)
library(tidyverse)
library(caret)

# Importar os dados (substitua com o caminho correto dos arquivos)
train_set <- read_excel("train_titanic.xlsx")
test_set  <- read_excel("test_titanic.xlsx")

# Verificar dados faltantes em ambos os conjuntos
list(train_set, test_set) %>% 
  map(~ colSums(is.na(.))) %>% 
  set_names(c("Train", "Test"))

# Adicionar coluna faltante 'Survived' ao conjunto de teste
test_set$Survived <- NA

# Adicionar coluna para identificar se o dado pertence ao conjunto de treino ou teste
train_set <- train_set %>% mutate(IsTrainSet = TRUE)
test_set  <- test_set  %>% mutate(IsTrainSet = FALSE)

titanic_set <- rbind(train_set, test_set)

# Transformações ETL/4Cs (Cleaning, Completing, Correcting, Creating)
titanic_set <- titanic_set %>% 
  mutate(
    across(c(Survived, Sex, Pclass, Embarked), as.factor),
    across(c(Age, SibSp, Parch, Fare), as.numeric),
    Fare = replace_na(Fare, median(Fare, na.rm = TRUE)),
    Age = replace_na(Age, median(Age, na.rm = TRUE)),
    Embarked = replace_na(Embarked, "S")
  )

# Separar conjuntos de treino e teste
titanic_train <- filter(titanic_set, IsTrainSet == TRUE)
titanic_test  <- filter(titanic_set, IsTrainSet == FALSE)

# Criar fórmula para o modelo
survived_formula <- Survived ~ Sex + Pclass + Age + SibSp + Parch + Fare + Embarked

# Treinar modelo de Random Forest
titanic_model <- randomForest(formula = survived_formula,
                              data = titanic_train,
                              ntree = 65,
                              importance = TRUE)

# Avaliar a importância das variáveis

vip::vip(titanic_model,
         mapping = aes(fill = .data[["Variable"]]),
         aesthetics = list(color = "grey35", size = 0.8))

# Fazer previsões no conjunto de treino para avaliar o desempenho do modelo
pred_train <- predict(titanic_model, newdata = titanic_train)

# Criar a matriz de confusão
cm <- confusionMatrix(pred_train, titanic_train$Survived)

# Imprimir a matriz de confusão
print(cm)

# Preparar dados para submissão
submission <- data.frame(
  PassengerId = titanic_test$PassengerId,
  Survived = predict(titanic_model, newdata = titanic_test)
)

# Salvar previsões em um arquivo CSV
write.csv(submission, file = "titanic_kaggle_r.csv", row.names = FALSE)

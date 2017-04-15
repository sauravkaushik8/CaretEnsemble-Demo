library('caretEnsemble')
library('caret')
set.seed(1)

data<-read.csv(url('https://datahack-prod.s3.ap-south-1.amazonaws.com/train_file/train_u6lujuX_CVtuZ9i.csv'),na.strings = '')

data$Gender[is.na(data$Gender)]<-'Male'
data$Married[is.na(data$Married)]<-'Yes'
data$Dependents[is.na(data$Dependents)]<-'0'
data$Self_Employed[is.na(data$Self_Employed)]<-'No'

#data<-data[complete.cases(data),]

prepro<-preProcess(data,method='medianImpute')

data<-predict(prepro,data)


index<-createDataPartition(data$Loan_Status,p=0.75,list = FALSE)

trainSet<-data[index,]
testSet<-data[-index,]

predictors<-c("Gender", "Married", "Dependents", "Education", 
              "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
              "Loan_Amount_Term", "Credit_History", "Property_Area")

outcomeName<-'Loan_Status'

control<-trainControl(method="cv",number = 5,savePredictions="final",classProbs=TRUE,index=createResample(trainSet$Loan_Status),summaryFunction=twoClassSummary)

model_list<-c("rpart","glm","gbm")

models<-caretList(trainSet[,predictors],trainSet[,outcomeName],methodList = model_list,metric='ROC',trControl = control)

models<-caretList(trainSet[,predictors],trainSet[,outcomeName],metric='ROC',trControl = control,  tuneList=list(
  rpart=caretModelSpec(method="rpart", tuneLength=3),
  rf=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=10), preProcess="pca"),
  gbm=caretModelSpec(method="gbm", tuneLength=3)
)
)

results <- resamples(models)
summary(results)
dotplot(results)

modelCor(results)




#Ensemble using GLM

greedy_ensemble <- caretEnsemble(
  models, 
  metric="ROC",
  trControl=trainControl(
    number=2,
    summaryFunction=twoClassSummary,
    classProbs=TRUE
  ))

summary(greedy_ensemble)
plot(greedy_ensemble)
varImp(greedy_ensemble)


#Stacking using GBM

gbm_ensemble <- caretStack(
  models,
  method="gbm",
  verbose=FALSE,
  tuneLength=10,
  metric="ROC",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)

summary(gbm_ensemble)
plot(gbm_ensemble)

setwd("C:/Users/Gourab/Downloads")

data <- read.csv('mnist_train.csv')

dim(data)

head(data[1:6])

unique(unlist(data[1]))

min(data[1:785])

max(data[1:785])

sample_2 <- matrix(as.numeric(data[6,-1]), nrow =28,byrow =TRUE)

image(sample_2 ,col =grey.colors((255)))



# rotating the image:

rotate<- function(x) t(apply(x,2,rev))  

image(rotate(sample_2), col =grey.colors(255))



#transform traget vars to factors:

is.factor(data$label)

data$label <- as.factor(data$label)

summary(data$label)

proportion <- prop.table (table(data$label))* 100

cbind(count = table(data$label) , proportion = proportion)



#distribution of pixels,eploratory analysis

#cental_block < c("pixel1376" , "pixel1337" , "pixel1404" , "pixel1405")

#par(mfrow = c(2,2))

#for (i in 1:9) {

# +hist(c(as.matrix(data[data$label==i , central_block])))













cran <- getOption("repos")

cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN?"

options(repos = cran)

install.packages("mxnet")



require(mxnet)

install.packages("ggplot2")

install.packages("caret")

train_perc <- 0.75

train_index<- createDataPartition(data$label,p=train_perc,list=FALSE)

data_train <- data[train_index,]

data_test <- data[-train_index,]

model_lr <- multinom(label ~. , data= data_train, MaxNWts = 10000, decay = 5e-3,maxit=100)

predicted_lr <- predict(model_lr , data_test ,type = "class")

cm_lr <- table(data_test$label,predicted_lr)

acc_lr = mean(predicted_lr == data_test$label)

accuracy_lr





model_nn <-nnet(label~.,data=data_train,size = 50,maxit = 100,MaxNWts = 100000, decay=1e-4)















data_train <- data.matrix(data_train)

data_train.x <- data_train[,-1]

data_train.x <- t(data_train.x/255)

data_train.y <- data_train[,1]



data<-mx.symbol.Variable("data")

#building the network:

# fisrt convolution layer



conv1 <- mx.symbol.Convolution(data = data, kernel=c(5,5),num_filter =20)

act1 <- mx.symbol.Activation(data = conv1 , act_type ="relu")

pool1 <-mx.symbol.Pooling(data = act1,pool_type= "max",kernel=c(2,2),stride=c(2,2))



# secod conv layer:



conv2 <- mx.symbol.Convolution(data=pool1,kernel =c(5,5),num_filter=50)

act2 <- mx.symbol.Activation(data = conv2,act_type ="relu")

pool2 <-mx.symbol.Pooling(data=act2, pool_type="max", kernel = c(2,2), stride = c(2,2))





#now we move on to the fully connected layer.but before that we need to flatten the resulting reduce maps from prevoius convolution layers:

flatten <- mx.symbol.Flatten(data=pool2)



#first fully connected layer,we apply two RElu hdden layers with 500 and 10 units respectively:

fc1 <-mx.symbol.FullyConnected(data = flatten,num_hidden=500)

act3 <-mx.symbol.Activation(data = fc1 ,act_type="relu")

#second fully connected layer

fc2<-mx.symbol.FullyConnected(data = act3 , num_hidden=10)



# Now the sofmax layer having output for 10 classes

softmax <- mx.symbol.SoftmaxOutput(data=fc2,name= "sm")



#now that the bone is created ,wehave to set a random seed and strat training

#first we need to reshape the matrix,data_train.x into an array as required by the mxnet package:

devices <- mx.cpu()

train.array <- data_train.x

dim(train.array) <- c(28,28,1,ncol(data_train.x))

mx.set.seed(42)

model_cnn <- mx.model.FeedForward.create(softmax,X=train.array,y=data_train.y,ctx=devices,num.round = 30,
                                         
                                         array.batch.size = 100,learning.rate =0.05,momentum=0.9,wd =0.00001,
                                         
                                         eval.metric = mx.metric.accuracy,epoch.end.callback= mx.callback.log.train.metric(100))



data1 <- read.csv('mnist_test.csv')

is.factor(data1$label)

data$label <- as.factor(data1$label)

summary(data1$label)

proportion <- prop.table (table(data1$label))* 100

cbind(count = table(data1$label) , proportion = proportion)





data_test <- data.matrix(data1)

data_test.x <- data_test[,-1]

data_test.x <- t(data_test.x/255)

data_test.y <- data_test[,1]



prob_dnn <- predict(model_cnn,data_test.x)

prediction_dnn <- max.col(t(prob_dnn)) -1

cm_dnn <- table(data_test$label,prediction_dnn)

cm_dnn





=================================================================================================================================================================================================================
  
  
  
  #Early Stoppage:
  
  validation_perc =0.4

validation_index <- createDaraPartition(data_test.y,p = validation_perc,list =FALSE)

validation.array <- test.array[,,,validation_index]

dim(validation.array) < c(28,28,1,length(validation.array[1,1,]))

data_validation.y<- data_test.y[validation_index]

final_test.array <- test.array[, , ,-validation_index]

dim(final_test.array) <- c(28,28,1,length(final_test.array[1,1,]))

data_final_test.y <- data_test.y[-validation_index]



#we write our custom callback function:

mx.callback.early.stop <- function(eval.metric){
  
  +function(iteration,nbatch,env,verbose){
    
    +if(!is.null(env$metric)){
      
      +if(!is.null(eval.metric)){
        
        +result <- env$metric$get(env$eval.metric)
        
        +if(result$value >= eval.metric){
          
          +return(FALSE)
          
          +}
        
        +}
      
      +}
    
    +return(TRUE)
    
    +}
  
  +}

#now we train the CNN model with early stopping,stoping criteria being greater than 0.985:

model_cnn_earlystop<-mx.model.FeedForward.ctreate(softmax,X=train.array,y=data_train.y,
                                                  
                                                  eval.data =list(data = validation.array,label=data_validation.y),
                                                  
                                                  +ctx=devices,num.round=30,array.batch.size=100,
                                                  
                                                  +learning.rate=0.05,momentum=0.9,wd=0.00001,eval.metric.accuracy,
                                                  
                                                  +epoch.end.callback =mx.callback.early.stop(0.985))





prob_cnn<-predict(model_cnn_earlystop,final_test.array)

prediction_cnn <-max.col(t(prob_cnn))-1

cm_cnn = table(data_final.y,prediction_cnn)

accuracy_cnn = mean(prediction_cnn == data-final_test.y)



================================================================================================================================================================================================================
  
  
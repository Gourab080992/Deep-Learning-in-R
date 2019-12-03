reticulate::use_python('/usr/bin/python3')

install.packages("devtools")

install.packages("Rcpp")

devtools::install_github("rstudio/keras")

library(keras)

devtools::install_github("rstudio/tensorflow")

library(tensorflow)

install_tensorflow()

setwd("C:/Users/Gourab/Downloads")

options(tensorflow.one_based_extract = FALSE)



##sample_2 <- matrix(as.numeric(mnist[1,5]), nrow =28,byrow =TRUE)

##image(sample_2 ,col =grey.colors((255)))





K<-keras::backend()

mnsist <-dataset_mnist()



X_train <- mnist$train$x

Y_train <-mnist$train$y

X_test <- mnist$test$x

Y_test <- mnist$test$y



par(mfrow = c(1,1))

plot(as.raster(X_train[8,,], max = 255))

par(mfrow= c(1,1))

X_train[3,,]

hist(X_train[1,,])







dim(X_train)<- c(nrow(X_train),784)

dim(X_test) <c(nrow(X_test),784)



X_train <- X_train/255

X_test <- X_test/255

X_train <- X_train %>% apply(1, as.numeric) %>% t()

X_test <- X_test %>% apply(1, as.numeric) %>% t()

class(X_test)



#WE construct the VAE,it consists of a latent dimension of size 2 and a hidden layer of 256 neurons.



orig_dim <-784

latent_dim <-2

inner_dim <-256

X<-layer_input(shape =c(orig_dim))

hidden_state<-layer_dense(X,inner_dim,activation="relu")

z_mean <- layer_dense(hidden_state,latent_dim)

z_log_sigma <- layer_dense(hidden_state,latent_dim)



#WE define the sampling function that generates sampling points from latent variables



sample_z <- function(params){
  
  z_mean <- params[,0:1]
  
  z_log_sigma <- params[,2:3]
  
  epsilon <- K$random_normal(
    
    shape = c(K$shape(z_mean)[[1]]),
    
    mean=0.,
    
    stddev = 1
    
  )
  
  z_mean + K$exp(z_log_sigma/2)*epsilon
  
}



#We define sample points:





z <- layer_concatenate(list(z_mean,z_log_sigma)) %>%
  
  layer_lambda(sample_z)



#WE define the encoder

decoder_hidden_state <- layer_dense(units = inner_dim , activation = "relu")

decoder_mean <- layer_dense(units=orig_dim,activation="sigmoid")

hidden_state_decoded <-decoder_hidden_state(z)

X_decoded_mean <- decoder_mean(hidden_state_decoded)





#VAE

variational_autoencoder <- keras_model(X,X_decoded_mean)

encoder <- keras_model(X,z_mean)

decoder_input <-layer_input(shape = latent_dim)

decoded_hidden_state_2 <- decoder_hidden_state(decoder_input)

decoder_X_mean_2 <- decoder_mean(decoded_hidden_state_2)

generator<- keras_model(decoder_input,decoder_X_mean_2)











# Custom loss function(KL- divergence penalisation)



loss_function <-function(X,decoded_X_mean){
  
  cross_entropy_loss <-loss_binary_crossentropy(X,decoded_X_mean)
  
  kl_loss<- -0.5*K$mean(1 +z_log_sigma - K$square(z_mean) -
                          
                          K$exp(z_log_sigma), axis = -1L)
  
  cross_entropy_loss +kl_loss}





















#COMPILATION:

variational_autoencoder %>% compile (optimizer ="rmsprop",loss=loss_function)

history<-variational_autoencoder%>% fit(X_train,X_train,shuffle=TRUE,epochs=10,batch_size=10,validation_data=list(X_test,X_test))

plot(history)







#RECONSTRUCTIONG THE TEST SET OUTPUT BY THE VAR

preds <-variational_autoencoder %>% predict(X_test)

dim(X_test)<- c(nrow(X_test),28,28)

dim(preds)<-c(nrow(preds),28,28)

Y_test[1]

image(255*X_test[1,,],col=gray.colors(3))



grid_x<-seq(-4,4,length.out=3)



#GENERATING DIGITS BY THE VARIATINAL AUTO ENCODER.



grid_x <-seq(-4,4,length.out =3)

grid_y <- seq(-4,4,length.out=3)

rows<-NULL

for(i in 1:length(grid_x)){
  
  column<-NULL
  
  for(j in 1:length(grid_y)){
    
    z_sample <- matrix(c(grid_x[i],grid_y[j],ncol=2)
                       
                       column>-rbind(column,predict(generator,z_sample)%>%matrix(ncol=28))
                       
  }
  
  rows <-cbind(rows,column)
  
}

rows %>%as.raster()%>%plot()
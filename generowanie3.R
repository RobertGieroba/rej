library(keras)
library(abind)
library(stringr)

maks_linii <- 5000
tekst <- tolower(paste0(readLines('rej.txt', maks_linii, encoding = 'UTF-8'), collapse='\n'))
tekst <- str_remove(tekst,"—")
tekst <- str_remove(tekst,"…")
dl <- 50
maks <- 3300

tokenizer <- text_tokenizer(num_words = maks)
tokenizer %>% fit_text_tokenizer(tekst)
ciag <- (tokenizer %>% texts_to_sequences(tekst))[[1]]
l_zn <- min(length(tokenizer$index_docs),maks)
litery <- (unlist(tokenizer$index_word, use.names = F))[1:maks]

dane <- array(0, c(length(ciag)-dl, dl+1))
for (i in 1:(length(ciag)-dl)){
  for (j in 1:dl){
    dane[i,] <- ciag[i:(i+50)]
  }
}
wz <- dane[,51]
dane <- dane[,1:50]
model <- keras_model_sequential()
model %>% layer_embedding(input_dim = l_zn, output_dim = 512, input_length = dl)
model %>% layer_lstm(256)
model %>% layer_dense(l_zn, 'softmax')
model %>% compile(optimizer_adam(lr=0.01), "sparse_categorical_crossentropy", list('acc'))
model %>% summary()
uczenie <- model %>% fit(dane,wz,epochs=10, view_metrics=F, batch_size=256, validation_split=0.15)
model %>% save_model_hdf5('generator')
nastepny_znak <- function(pred, temperatura = 1.0) {
  pred <- as.numeric(pred)
  pred <- log(pred) / temperatura
  exp_pred <- exp(pred)
  pred <- exp_pred / sum(exp_pred)
  which.max(t(rmultinom(1, 1, pred)))
}
jedynka <- function(dl, i){
  c(rep(0,i-1),1,rep(0, dl-i))
}

generuj_tekst <- function(start, ile, temp){
  wynik <- rep(0, ile)
  d <- array(start, c(1, dl))
  for(i in 1:ile){
    pred <- (model %>% predict(d))[1,]
    znak <- nastepny_znak(pred, temp)
    d <- cbind(d[,2:ncol(d), drop=F], znak)
    wynik[i] <- znak
  }
  paste(litery[wynik], collapse = ' ')
}

start <- dane[1,]
a <- generuj_tekst(start, 250, 0.75)
cat(a)


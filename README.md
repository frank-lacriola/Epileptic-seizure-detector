La tecnica proposta riguarda un algoritmo di deep learning in grado di classificare con la massima accuratezza possibile se un paziente sia un soggetto epilettico o meno, 
a partire dai dati delle registrazioni dell’elettroencefalogramma a lungo termine (EEG) a cui il paziente è sottoposto, un esame diagnostico che, attraverso alcuni elettrodi
posizionati sul cuoio capelluto, misura l‘attività elettrica cerebrale.
Con un approccio semi-supervisionato, l’algoritmo consta di una prima parte in cui è implementato un Autoencoder (tecnica di apprendimento automatico con approccio non 
supervisionato), ovvero una rete neurale che in questo caso è utilizzata per effettuare una feature reduction, ovvero un processo con lo scopo di ridurre la dimensionalità 
della grande mole di dati forniti dall’EEG e quindi dal data set. In seguito, il data set ridotto viene dato in input alla seconda parte dell’algoritmo che presenta un modello 
di rete neurale atta alla classificazione (tecnica di apprendimento automatico con approccio supervisionato) dei gruppi di dati che le vengono forniti.

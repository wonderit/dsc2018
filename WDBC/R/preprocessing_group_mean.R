library('dplyr')

#전체 데이터 group mean 추가
csv<-read.csv("./data/speeddating_preprocessed_id_DA.csv")

data <- csv %>%
  round() %>%
  group_by(iid) %>%
  mutate(att_m = round(mean(attractive_o),1)) %>%
  mutate(sin_m = round(mean(sinsere_o),1)) %>%
  mutate(int_m = round(mean(intelligence_o),1)) %>%
  mutate(fun_m = round(mean(funny_o),1)) %>%
  mutate(amb_m = round(mean(ambitous_o),1)) %>%
  mutate(sha_m = round(mean(shared_interests_o),1)) %>%
  mutate(dec_m = round(mean(decision_o),1)) %>%
  select(-decision_o,decision_o)

write.csv(data,"./data/speeddating_preprocessed_id_mean.csv",row.names = FALSE)

#train 데이터 group mean 추가
csv<-read.csv("./data/speeddating_preprocessed_id_test.csv")

data <- csv %>%
  round() %>%
  group_by(iid) %>%
  mutate(att_m = round(mean(attractive_o),1)) %>%
  mutate(sin_m = round(mean(sinsere_o),1)) %>%
  mutate(int_m = round(mean(intelligence_o),1)) %>%
  mutate(fun_m = round(mean(funny_o),1)) %>%
  mutate(amb_m = round(mean(ambitous_o),1)) %>%
  mutate(sha_m = round(mean(shared_interests_o),1)) %>%
  mutate(dec_m = round(mean(decision_o),1)) %>%
  select(-decision_o,decision_o)

write.csv(data,"./data/speeddating_preprocessed_id_mean_test.csv",row.names = FALSE)

#test 데이터 group mean 추가
csv<-read.csv("./data/speeddating_preprocessed_id_train.csv")

data <- csv %>%
  round() %>%
  group_by(iid) %>%
  mutate(att_m = round(mean(attractive_o),1)) %>%
  mutate(sin_m = round(mean(sinsere_o),1)) %>%
  mutate(int_m = round(mean(intelligence_o),1)) %>%
  mutate(fun_m = round(mean(funny_o),1)) %>%
  mutate(amb_m = round(mean(ambitous_o),1)) %>%
  mutate(sha_m = round(mean(shared_interests_o),1)) %>%
  mutate(dec_m = round(mean(decision_o),1)) %>%
  select(-decision_o,decision_o)

write.csv(data,"./data/speeddating_preprocessed_id_mean_train.csv",row.names = FALSE)

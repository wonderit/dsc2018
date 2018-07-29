library('ggplot2')
library('dplyr')
library('corrplot')

#데이터 로드 및 전처리
csv<-rbind(read.csv("./data/speeddating_likeo_train_yj.csv"),read.csv("./data/speeddating_likeo_test_yj.csv"))
prob <- read.csv("./data/speeddating_id.csv", na.strings = '?') %>%
  select(c('iid','pid','guess_prob_liked')) %>%
  na.omit()

data <- inner_join(csv,prob) %>%
  select(-c(1,2,5,8:25,31:36,46,47))

#corrplot
corrplot.mixed(cor(data),upper = 'number',lower = 'circle',tl.pos = 'lt')
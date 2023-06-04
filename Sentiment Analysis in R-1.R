knitr::opts_chunk$set(echo = TRUE,message = FALSE, warning = FALSE)


# Load up the libraries and dataset
library(tidyverse)
library(sentimentr)
library(caret)
library(quanteda)
library(broom)

# Load up the .CSV data and explore in RStudio.
hotel_raw <- read_csv("Data/hotel-reviews.csv")
set.seed(1234)# we set seed to replicate our results.
hotel_raw<-hotel_raw[sample(nrow(hotel_raw), 5000), ]# take a small sample
corp_hotel <- corpus(hotel_raw, text_field = "Description")# Create corpus
sample<-corpus_sample(corp_hotel, size = 20)# you can sample a corpus too!


library(quanteda)
test.lexicon <- dictionary(list(positive.terms = c("happy", "joy", "light"),
                                negative.terms = c("sad", "angry", "darkness")))

testtext<-c("I am happy and confident that the paper will be accepted", 
            "Of course, no one can be 100% sure but I am hopeful",
            "In case, it is rejected, I will be sad and angry, we will submit it to another journal")

dfm_sentiment1 <- testtext %>% tokens() %>% dfm() %>% dfm_lookup(test.lexicon)
dfm_sentiment1

positive_words_bing <- scan("Data/positive-words.txt", what = "char", sep = "\n", skip = 35, quiet = T)
negative_words_bing <- scan("Data/negative-words.txt", what = "char", sep = "\n", skip = 35, quiet = T)
sentiment_bing <- dictionary(list(positive = positive_words_bing, negative = negative_words_bing))

dfm_sentiment <- corp_hotel %>% tokens() %>% dfm %>% dfm_lookup(sentiment_bing)
dfm_sentiment
dfm_sentiment_df<-data.frame(dfm_sentiment)
dfm_sentiment_df$net<-(dfm_sentiment_df$positive)-(dfm_sentiment_df$negative)
summary(dfm_sentiment_df)# document level summary

## install.packages("remotes")
## remotes::install_github("kbenoit/quanteda.dictionaries")

library("quanteda.dictionaries")
output_mfd <- quanteda.dictionaries::liwcalike(corp_hotel, 
                        dictionary = data_dictionary_MFD)
head(output_mfd)

# Proportions instead of numbers

dfm_sentiment_prop <- dfm_weight(dfm_sentiment, scheme = "prop")
dfm_sentiment_prop

## Plotting the sentiments

sentiment <- convert(dfm_sentiment_prop, "data.frame") %>%
    gather(positive, negative, key = "Polarity", value = "Share") %>% 
    mutate(document = as_factor(doc_id)) %>% 
    rename(Review = document)

ggplot(sentiment, aes(Review, Share, fill = Polarity, group = Polarity)) + 
    geom_bar(stat='identity', position = position_dodge(), size = 1) + 
    scale_fill_brewer(palette = "Set1") + 
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) + 
    ggtitle("Sentiment scores in Hotel Reviews (relative)")


mytext<-"I am happy and confident that the paper will be accepted.
Of course, no one can be 100% sure but I am hopeful. In case, it is rejected,
I will be sad and angry, but we will submit it to another journal."

mytext <- get_sentences(mytext)
sentiment(mytext)

mytext2<-"I am happy and confident that the paper will be accepted.
Of course, no one can be 100% sure but I am hopeful. In case, it is rejected,
I will not be sad and angry, but we will submit it to another journal." # with a negator added
mytext2 <- get_sentences(mytext2)
sentiment(mytext2)


out <- with(
    hotel_raw, 
    sentiment_by(
        get_sentences(Description), # Reviews are stored in variable Description
        list(User_ID,Device_Used) # grouping variables
    ))
head(out)

library(magrittr)
library(dplyr)
set.seed(234)

hotel_raw %>%
    filter(User_ID %in% sample(unique(User_ID), 4)) %>% 
    # %in% operator in R, is used to identify if an element belongs to a vector.
    mutate(review = get_sentences(Description)) %$%
    # The “exposition” pipe operator from magrittr package, %$% exposes the names  
    # within the left-hand side object to the right-hand side expression. For  
    # instance,
    # iris %>%
    # subset(Sepal.Length > mean(Sepal.Length)) %$%
    # cor(Sepal.Length, Sepal.Width)
    sentiment_by(review, User_ID) %>%
    highlight()


library(tidyverse)
library(cleanNLP)

cnlp_init_udpipe() # Loading namespace: udpipe which serves backend to cleanNLP

hotel_raw1<-hotel_raw[sample(nrow(hotel_raw), 100), ]# take a small sample as POS 
                                              #tagging is very resource intensive

postag<-hotel_raw1 %>% 
cnlp_annotate(text= "Description") # Outcome is list of tokens and documents

head(postag$token,n=10)

postag$token %>%
  filter(xpos == "JJ") %>% # you can play around with different POS
  group_by(lemma) %>%
  summarize(count = n()) %>%
  top_n(n = 10, count) %>%
  arrange(desc(count)) 

postag$token %>%
  group_by(doc_id) %>%
  summarize(n = n()) %>%
  left_join(postag$document, by="doc_id") %>%
  ggplot(aes(Device_Used, n)) +
    geom_line(color = grey(0.8)) +
    geom_point(aes(color = Is_Response)) +
    geom_smooth(method="loess", formula = y ~ x) +
    theme_minimal()

require(quanteda)
require(quanteda.textmodels)
require(caret)

summary(corp_hotel, 5)# let's check the summary of our original corpus

# generate 3500 numbers without replacement
set.seed(300)
id_train <- sample(1:5000, 3500, replace = FALSE)

# create docvar with ID
corp_hotel$id_numeric <- 1:ndoc(corp_hotel)

# get training set
dfmat_training <- corpus_subset(corp_hotel, id_numeric %in% id_train) %>%
    dfm(remove = stopwords("english"), stem = TRUE)

#Since we will run  the binary multinomial NB model, let's convert the dfm to a binary matrix before training the model. 

dfmat_training<-dfm_weight(dfmat_training, scheme = "boolean")

# get test set (documents not in id_train)
dfmat_test <- corpus_subset(corp_hotel, !id_numeric %in% id_train) %>%
    dfm(remove = stopwords("english"), stem = TRUE)

dfmat_test<-dfm_weight(dfmat_test, scheme = "boolean") 

tmod_nb <- textmodel_nb(dfmat_training, dfmat_training$Is_Response)
summary(tmod_nb)


dfmat_matched <- dfm_match(dfmat_test, features = featnames(dfmat_training))


actual_class <- dfmat_matched$Is_Response
predicted_class <- predict(tmod_nb, newdata = dfmat_matched)
tab_class <- table(actual_class, predicted_class)
tab_class

#confusionMatrix(tab_class, mode = "everything")



# Load up the .CSV data and explore in RStudio.
hotel_raw <- read_csv("Data/hotel-reviews.csv")

hotel_raw <- hotel_raw %>%
    mutate(hotel_name = case_when(str_detect(Description, "Hilton")~ "Hilton",
                                  str_detect(Description, "Hyatt")~ "Hyatt",
                                  str_detect(Description, "Marriott")~ "Marriott"))
hotel_sub <- hotel_raw %>% filter(hotel_name=="Hilton"|hotel_name=="Hyatt"|hotel_name=="Marriott")

# Create a new column rating based on a string condition
hotel_sub <- hotel_sub %>%
        mutate(rating = ifelse(str_detect(hotel_name, "Marriott"), 
        round(rnorm(nrow(filter(hotel_sub, str_detect(hotel_name, "Marriott"))),
                    mean = 4.2, sd = .25)), NA)) %>%
        mutate(rating = ifelse(str_detect(hotel_name, "Hilton"), 
        round(rnorm(nrow(filter(hotel_sub, str_detect(hotel_name, "Hilton"))),
                    mean = 3.55, sd = .33)), rating)) %>%
        mutate(rating = ifelse(str_detect(hotel_name, "Hyatt"), 
        round(rnorm(nrow(filter(hotel_sub, str_detect(hotel_name, "Hyatt"))),
                    mean = 3.25, sd = .35)), rating))


abc <- hotel_sub %>%
    get_sentences('Description') %>%
    sentiment_by(by = 'User_ID')

joined_data <- inner_join(hotel_sub, abc, by="User_ID")

# Perform an ANOVA
model <- aov(rating ~ as.factor(hotel_name), data = joined_data)

# Print the summary of the model
summary(model)

# Calculate the means for each group
group_means <- aggregate(rating ~ as.factor(hotel_name), hotel_sub, mean)

# Print the group means
print(group_means)

# Perform a Tukey's HSD test
posthoc <- TukeyHSD(model)

# Print the results of the post-hoc test
print(posthoc)


# Run a linear regression
model1 <- lm(rating ~ Is_Response+ hotel_name +word_count, data = joined_data)

# Print the summary of the model
summary(model1)


# Run a linear regression
model2 <- lm(rating ~ Is_Response+ hotel_name+word_count +ave_sentiment, data = joined_data)

# Print the summary of the model
summary(model2)



library(tidyverse)

reviews<-read_csv("amazon_reviews.csv")

library(quanteda)
test.lexicon <- dictionary(list(positive.terms = c("happy", "excited", "great","easy","glad"),
                                negative.terms = c("disappointed", "terrible", "reset", "reboots", "annoyed")))

dfm_sentiment1 <- reviews$text %>% tokens() %>% dfm() %>% dfm_lookup(dictionary = test.lexicon)
dfm_sentiment1

dfm_sentiment_df<-convert(dfm_sentiment1, 'data.frame')
dfm_sentiment_df$net<-(dfm_sentiment_df$positive)-(dfm_sentiment_df$negative)
summary(dfm_sentiment_df)# document level summary

################################################

library("quanteda.dictionaries")
output_mfd <- liwcalike(reviews$text,
                        dictionary = data_dictionary_MFD)
head(output_mfd)
#########################################################

# Proportions instead of numbers

dfm_sentiment_prop <- dfm_weight(dfm_sentiment1, scheme = "prop")
dfm_sentiment_prop

## Plotting the sentiments


sentiment <- convert(dfm_sentiment_prop, "data.frame") %>%
    gather(positive.terms, negative.terms, key = "Polarity", value = "Share") %>% 
    mutate(document = as_factor(doc_id)) %>% 
    rename(Review = document)

senti20<-sample_n(sentiment, 20)
ggplot(senti20, aes(Review, Share, fill = Polarity, group = Polarity)) + 
    geom_bar(stat='identity', position = position_dodge(), size = 1) + 
    scale_fill_brewer(palette = "Set1") + 
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) + 
    ggtitle("Sentiment scores in Hotel Reviews (relative)")


########################################

library(sentimentr)
mytext <- get_sentences(reviews$text)
head(sentiment(mytext))


## By sentence sentiment for each rating category
out <- with(
    reviews, 
    sentiment_by(
        get_sentences(text), # Reviews are stored in variable text
        list(rating) # grouping variables
    ))
head(out)

#### We want to find out what brands are mentioned in the review context (Use nouns for feature level analysis)

library(tidyverse)
library(cleanNLP)

cnlp_init_udpipe() # Loading namespace: udpipe which serves backend to cleanNLP

reviews_sample<-sample_n(reviews,100)
# take a small sample as POS 
#tagging is very resource intensive

postag<-reviews_sample %>% 
    cnlp_annotate(text= "text") # Outcome is list of tokens and documents

head(postag$token,n=10)

postag$token %>%
    filter(xpos == "NNP") %>% # you can play around with different POS
    group_by(lemma) %>%
    summarize(count = n()) %>%
    top_n(n = 10, count) %>%
    arrange(desc(count)) 





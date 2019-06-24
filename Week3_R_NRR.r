
# Setting directory and importing data
setwd('C:\\Users\\nrrvlkp\\Documents\\M\\630\\Projects\\Predictive_analytics\\data')

# importing required libraries
library(ggplot2)
library(Hmisc)
library(caTools)

# Reading data
data <- read.table('dodgers.csv' , header=TRUE, sep = ',')
head(data)

# print summary data
summary(data)



# check for null values in data frame
any(is.na(data))

# print structure of data 
str(data)

# number of rows
nrow(data)

# number of columns
ncol(data)

# Histogram with Frequency
hsplot <- ggplot(data, aes(x= data$attend)) +
  geom_histogram(binwidth= 1000) +
  theme_classic()+ 
  labs(x = "Attendance", y= "Frequency", title = "Histogram for attend")

hsplot

dnplot_attend <- ggplot(data, aes(x= data$attend)) +
  geom_density(fill = "skyblue") +
  theme_classic()+ 
  labs(x = "Attendance", y= "Density", title = "Density plot for Attendance")

dnplot_attend

# Density plot to check distribution of temperature
dnsplot_temp <- ggplot(data, aes(x= data$temp,  y=..density..)) +
  geom_density(fill = "lightgreen") +
  #geom_point() +
  labs(x = "Temperature", y= "Density", title = "Density plot of Temparature")

dnsplot_temp

# examine the data through descriptive statistics
Hmisc::describe(data)

# Box plot to check outlier for attendance

bxplot_wkday <- ggplot(data, aes(x = day_of_week , y = data$attend)) +
  geom_boxplot(aes(fill = day_of_week))+
  theme_bw()+
  labs(x = "Week Day", y= "Attendence", title = "Box plot for attendance for each week day")

bxplot_wkday

bxplot_month <- ggplot(data, aes(x = month , y = data$attend)) +
  geom_boxplot(aes(fill = month))+
  theme_bw()+
  labs(x = "Month", y= "Attendence", title = "Box plot for attendance for each month")

bxplot_month

violinplot <- ggplot(data, aes(x = day_of_week, y = data$attend))+
  geom_violin(aes(fill = day_of_week))+
  geom_jitter(height = 0, width = 0.1)+
  stat_summary(fun.y = mean, geom = "point", size = 3, color = "red")+ # adding mean
  theme_bw()+
  labs(x = "Week Day", y= "Attendence", title = "Violin plot for attendance")

violinplot

# Looking at bar plots to see how the attendence is for each day of week

attend_by_wkday %>% select(data, day_of_week, attend)
attend_by_wkday

grp_wkday <- aggregate(attend_by_wkday$attend, by = list(attend_by_wkday$day_of_week), FUN = (sum))
grp_wkday

ggplot(data = data, aes(x=day_of_week, y = attend))+ 
  geom_bar(stat="identity", fill = "steelblue")+
  labs(title = "Bar plot of day of week Vs Attendance", y = "Attendence", x = "Day of week")


stackedbar <- ggplot(data = data, aes(x=day_of_week, y = attend, fill = factor(day_night)))+ 
  geom_bar(stat="identity") +
  labs(title = "Bar plot of day of week Vs Attendance", y = "Attendence", x = "Day of week")
stackedbar

# plotting attendance with respective weather 

stackedbar_wthr <- ggplot(data = data, aes(x=day_of_week, y = attend, fill = factor(skies)))+ 
  geom_bar(stat="identity") +
  labs(title = "Bar plot of day of week Vs Attendance", y = "Attendence", x = "Day of week")
stackedbar_wthr

# plotting attendance with respective weather 

stackedbar_mth <- ggplot(data = data, aes(x=month, y = attend, fill = factor(skies)))+ 
  geom_bar(stat="identity") +
  labs(title = "Bar plot of day of week Vs Attendance", y = "Attendence", x = "Month")
stackedbar_mth


# plotting attendance with respective weather 
stackedbar_opp <- ggplot(data = data, aes(x=opponent, y = attend, fill = factor(day_of_week)))+ 
  geom_bar(stat="identity") +
  coord_flip()+
  labs(title = "Bar plot of day of week Vs Attendance", y = "Attendence", x = "Opponent")
stackedbar_opp

# Bar plots with Opponents

ggplot(data = data)+ 
  geom_histogram(stat = "count", mapping = aes(x = opponent), fill = "steelblue")+
  coord_flip()+
  labs(title = "Number of matches by Opponent", y = "Counts", x = "Opponent")


ggplot(data=data, aes(x=attend, y=temp)) + geom_point()+
  theme_bw() +
  geom_smooth(method=lm, se=FALSE) +
  labs(title = "Scatter plot between attend and temp", x = "Attendence", y = "Temparature")

# looking at scatter plot with respetive other variable - day of week
ggplot(data=data, aes(x=attend, y=temp)) + geom_point()+
  theme_bw() +
  geom_smooth(method=lm, se=FALSE) +
  facet_wrap(~day_night)+
  labs(title = "Scatter plot between attend and temp", x = "Attendence", y = "Temparature")

# Scatter plot for attendence by day of week
ggplot(data=data, aes(x=attend, y=temp, color = day_of_week)) + geom_point()+
  theme_bw() +  # geom_smooth(method=loess, se=FALSE)
labs(title = "Scatter plot between attend and temp", x = "Attendence", y = "Temparature")



oneway.test(attend ~ day_of_week, data = data)

oneway.test(attend ~ month, data = data)

oneway.test(attend ~ opponent, data = data)

oneway.test(attend ~ day_night, data = data)

cor(data$attend, data$temp) 

oneway.test(attend ~ skies, data = data)

oneway.test(attend ~ cap, data = data)

oneway.test(attend ~ shirt, data = data)

oneway.test(attend ~ fireworks, data = data)

oneway.test(attend ~ bobblehead, data = data)

# splitting data into test and train
require(caTools)
 
#for consistent results
set.seed(1000)

sample = sample.split(data, SplitRatio = 0.75)

train = subset(data, sample == TRUE)
nrow(train)

test = subset(data, sample == FALSE)
nrow(test)

fn.model1 = {attend ~ day_of_week  + bobblehead }

mlm.fit.train1 = lm(fn.model1, data = train)
summary(mlm.fit.train1)

summary(aov(mlm.fit.train1))

fn.model2 = {attend ~ day_of_week + month + bobblehead }

mlm.fit.train2 = lm(fn.model2, data = train)
summary(mlm.fit.train2)



summary(aov(mlm.fit.train2))

fn.model3 = {attend ~  day_of_week + month + bobblehead + opponent}

mlm.fit.train3 = lm(fn.model3, data = train )
summary(mlm.fit.train3)


summary(aov(mlm.fit.train3))



# plotting density plot for residuals of the model
plot(density(mlm.fit.train1$residuals))

#plotting model
op = par(mfrow = c(2, 2))
plot(mlm.fit.train1)


# plotting density plot for residuals of the model
plot(density(mlm.fit.train2$residuals))

#plotting model
op = par(mfrow = c(2, 2))
plot(mlm.fit.train2)





# predicting from training data set
train$predict_attend <- predict(mlm.fit.train2)
train$predict_attend


# predicting from test data set
test$predict_attend <- predict(mlm.fit.train2, newdata = test)

test$error = test$predict_attend - test$attend

test

ggplot(data=test, aes(x=predict_attend, y=error)) + geom_point()+
  theme_bw() +
  geom_smooth(aes(x=predict_attend, y=error)) +#method=lm, se=FALSE) +
  labs(title = "Scatter plot between Attendence and Error from prediction", x = "Attendance", y = "Error")



# predicting from test data set using Model1 (model1 = {attend ~ day_of_week  + bobblehead })
test$predict_attend1 <- predict(mlm.fit.train1, newdata = test)
test$error1 = test$predict_attend1 - test$attend

ggplot(data=test, aes(x=predict_attend1, y=error1)) + geom_point()+
  theme_bw() +
  geom_smooth(aes(x=predict_attend1, y=error1)) +#method=lm, se=FALSE) +
  labs(title = "Scatter plot between Attendence and Error from prediction (using model1)", x = "Attendance", y = "Error")



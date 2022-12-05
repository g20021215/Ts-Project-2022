A <- read.csv("Yantai_Short.csv",header = TRUE)
y <- A$AverageTemperature
y <- ts(y,start = 1970,frequency = 12)
y
a=diff(y, lag=12)# 先去季节项
getwd()
x <- seq(1,length(a))
lmxy <- lm(a~x)
x11()
par(mfrow=c(2,2)) #2 x 2 output for plots
resid<-resid(lmxy)
fit<-fitted(lmxy)
qqnorm(resid,main=NA,xlab='Theroretical',ylab='Sample Quantiles') #normal plot
qqline(resid,col="red")
title(sub='(a)')
plot(fit,resid,xlab='Fitted value', ylab='Residual') #residual vs. fitted value plot
points(fit,rep(0, n),type='l')
title(sub='(b)')
hist(resid,xlab='Residual', main=NA) #histogram
title(sub='(c)')
plot(x,resid,type='l',xlab='Observation order',ylab='Residual') #residual vs. observation order (t) plot
points(x,resid,pch=19)
points(x,rep(0, n),type='l')
title(sub='(d)')
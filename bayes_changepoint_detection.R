library("changepoint")
df <- read.csv(file = 'tot_Monthly_Metrics.csv')
y = df$'Underutilization'

#library(boot)
#data(coal)
#y <- tabulate(floor(coal[[1]]))
#y <- y[1851:length(y)]

barplot(y, xlab="years", ylab="Underutilization")
results <- cpt.mean(y,method="AMOC")
cpts(results)
param.est(results)
plot(results,cpt.col="blue",xlab="Index",cpt.width=4)

barplot(y, xlab="years", ylab="Overutilization")
results <- cpt.mean(y,method="AMOC")
cpts(results)
param.est(results)
plot(results,cpt.col="blue",xlab="Index",cpt.width=4)

barplot(y, xlab="years", ylab="Productivity")
results <- cpt.mean(y,method="AMOC")
cpts(results)
param.est(results)
plot(results,cpt.col="blue",xlab="Index",cpt.width=4)
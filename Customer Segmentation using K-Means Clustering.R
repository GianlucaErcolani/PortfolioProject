## Data Import and Exploration
Customers=read.csv("C:/Users/gianl/Downloads/archive (2)/Mall_Customers.csv")

head(Customers)

summary(Customers[,2:5])

#Checking missing values
sum(is.na(Customers))


hist(Customers$Age,col='red', main='Histogram of Age', xlab='Age')
hist(Customers$Annual.Income..k..,col='blue',main='Histogram of Annual income', xlab='Annual income')
hist(Customers$Spending.Score..1.100.,col='green', main='Histogram of Spending score', xlab='Spending score')

#equal number of bins (10)
hist(Customers$Age,col='red', main='Histogram of Age', xlab='Age', breaks = 10)
hist(Customers$Annual.Income..k..,col='blue',main='Histogram of Annual income', xlab='Annual income', breaks = 10)
hist(Customers$Spending.Score..1.100.,col='green', main='Histogram of Spending score', xlab='Spending score', breaks = 10)


#data exploration with ggplot2
#numeric variables
library(ggplot2)

ggplot(Customers, aes(Age))+geom_histogram(color="red", fill="red",bins=10,alpha=.2)+
        geom_histogram(aes(Annual.Income..k..), color="blue", fill="blue",bins=10,alpha=.2)+
        labs(title='Histograms of Age and Annual.Income..k..', x='Age (blue), Annual.Income..k.. (red)')

ggplot(Customers, aes(Age))+geom_histogram(color="red", fill="red",bins=10,alpha=.2)+
        geom_histogram(aes(Spending.Score..1.100.), color="green", fill="green",bins=10,alpha=.2)+
        labs(title='Histograms of Age and Spending.Score..1.100.', x='Age (blue), Spending.Score..1.100. (red)')

ggplot(Customers, aes(Spending.Score..1.100.))+geom_histogram(color="green", fill="green",bins=10,alpha=.2)+
        geom_histogram(aes(Annual.Income..k..), color="blue", fill="blue",bins=10,alpha=.2)+
        labs(title='Histograms of Spending.Score..1.100. and Annual.Income..k..', x='Spending.Score..1.100. (blue), Annual.Income..k.. (red)')

#categorical variables
ggplot(Customers, aes(Gender)) +geom_bar()

library(dplyr) 
Customers %>% count(Gender)

# Scaling
New=Customers
New[,3:5]=scale(New[,3:5]) # we scale the columns of interest only
head(New)


# Choosing the number of clusters
# Inertia

library(cluster)
library(factoextra)

# Define the wss function
wss=function(k) {
        kmeans(New[,4:5],centers=k,nstart=30)$tot.withinss 
}

# Set the number of k from 1 to 10
kvalues=1:10

# Save the inertia information in
inertia=c()
for (k in kvalues) {
        inertia[k]=wss(k)   
}

# Create a simple plot
plot(kvalues, inertia,
     type="b", pch = 10, col=2, frame = FALSE, 
     xlab="Number of clusters k",
     ylab="Inertia")

# inertia using the function "fviz_nbclust"
fviz_nbclust(New[,4:5], kmeans, method = "wss")

# Silhouette
fviz_nbclust(New[,4:5], kmeans, method = "silhouette")

# The gap method
gap=clusGap(New[,4:5], kmeans, nstart = 30,
            K.max = 10, B = 100) # B is number of bootstrap samples
print(gap)

fviz_gap_stat(gap)


# The K-Means Algorithm
km.out=kmeans(New[,4:5],centers=5,nstart=30)

km.out$centers
km.out$size
km.out$cluster

New$Cluster=km.out$cluster
head(New)


# Plotting
ggplot(New, aes(Annual.Income..k.., Spending.Score..1.100.))+geom_point()+
        labs(title='Scatter plot of Annual income and Spending Score',x='Annual income', y='Spending Score')

#different colours for different clusters
ggplot(New, aes(Annual.Income..k.., Spending.Score..1.100.))+geom_point(color=km.out$cluster)+
        labs(title='K-means clustering of customers segmentation',x='Annual income', y='Spending Score')

#data frame containing the centroids
centroids = data.frame(Annual.Income..k.. =km.out$centers[,1], Spending.Score..1.100. =km.out$centers[,2])

#add centroids as little blue triangles
ggplot(New, aes(Annual.Income..k.., Spending.Score..1.100.))+
        geom_point(color=km.out$cluster)+ geom_point(data=centroids, color ='blue',shape=17)+
        labs(title='K-means clustering of customers segmentation',x='Annual income', y='Spending Score',
             color='Group')

#Add CustomerID info
ggplot(New, aes(Annual.Income..k.., Spending.Score..1.100.))+
        geom_point(color=km.out$cluster)+ geom_point(data=centroids, color ='blue',shape=17)+
        geom_text(label=New$CustomerID,nudge_x = 0.08,size=2,check_overlap = T)+
        labs(title='K-means clustering of customers segmentation',x='Annual income', y='Spending Score',
             color='Group')


















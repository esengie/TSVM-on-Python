titanic = rbind(t(replicate(35,c(0,0,1,0,  0,0,0))), t(replicate(17,c(0,0,1,0,  1,0,0))), 
	t(replicate(118,c(1,0,0,0,  0,1,0))), t(replicate(154,c(0,1,0,0,  0,1,0))), t(replicate(387,c(0,0,1,0,  0,1,0))), t(replicate(670,c(0,0,0,1,  0,1,0))), 
	t(replicate(4,c(1,0,0,0,  1,1,0))), t(replicate(13,c(0,1,0,0,  1,1,0))),t(replicate(89,c(0,0,1,0,  1,1,0))), t(replicate(3,c(0,0,0,1,  1,1,0))), 

	t(replicate(5,c(1,0,0,0,  0,0,1))), t(replicate(11,c(0,1,0,0,  0,0,1))),t(replicate(13,c(0,0,1,0,  0,0,1))), 
	t(replicate(1,c(1,0,0,0,  1,0,1))), t(replicate(13,c(0,1,0,0,  1,0,1))),t(replicate(14,c(0,0,1,0,  1,0,1))),
	t(replicate(57,c(1,0,0,0,  0,1,1))), t(replicate(14,c(0,1,0,0,  0,1,1))),t(replicate(75,c(0,0,1,0,  0,1,1))), t(replicate(192,c(0,0,0,1,  0,1,1))), 
	t(replicate(140,c(1,0,0,0,  1,1,1))), t(replicate(80,c(0,1,0,0,  1,1,1))),t(replicate(76,c(0,0,1,0,  1,1,1))), t(replicate(20,c(0,0,0,1,  1,1,1))))
	
miniM = rbinom(dim(titanic)[1], size = 1, prob = 0.7)
train = titanic[miniM,]
test = titanic[!miniM,]

write.table(train[,1:6], file = "train_samples.csv", row.names = F, col.names = F, sep=',')
write.table(train[,7], file = "train_answers.csv", row.names = F, col.names = F, sep=',')

write.table(test[,1:6], file = "test_samples.csv", row.names = F, col.names = F, sep=',')
write.table(test[,7], file = "test_answers.csv", row.names = F, col.names = F, sep=',')
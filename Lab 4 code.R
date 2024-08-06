install.packages("tidyverse")
library(tidyverse)
View(Class_Spring_2022_1)
ggplot(data = Class_Spring_2022_1 , aes(x = Sleep)) +
  geom_density(fill = "green", color = "black") +
  labs(title = "Sleep Graph")
mean(Class_Spring_2022_1$Sleep)
sd(Class_Spring_2022_1$Sleep)
ggplot(data = Class_Spring_2022_1, aes(x = Coffee,  y = Sleep, fill = Coffee)) + 
  geom_boxplot(color = "black") +
  stat_boxplot(geom = "errorbar") +
  labs(title = "Coffee drinkers vs. Non Coffee Drinkers")
Class_Spring_2022_1$Academ_Level = factor(Class_Spring_2022_1$Academ_Level, 
                                        levels = c("Freshman", "Sophomore", "Junior", "Senior"))
ggplot(data = Class_Spring_2022_1, aes(x = Academ_Level, y = Sleep, color = Academ_Level)) +
  geom_jitter(width = 0.1) + 
  labs(title = "Sleep By Year")
fresh_data = subset(Class_Spring_2022_1, Academ_Level == "Freshman")
nrow(subset(Class_Spring_2022_1, Academ_Level == "Freshman"))
summary(subset(Class_Spring_2022_1, Academ_Level == "Freshman"))
summary(subset(Class_Spring_2022_1, Academ_Level == "Junior"))

ggplot(data = Class_Spring_2022_1, aes(x = BPM, y = Sleep, color = Academ_Level)) +
  geom_point(alpha = 0.25) +
  labs(title = "Sleep and BPM")
ggplot(data = Class_Spring_2022_1, aes(x = Academ_Level, fill = Car)) + 
  geom_bar(position = "fill", color = "white") +
  labs(title = "Car Ownership by Grade")
ggplot(data = Class_Spring_2022_1, aes(x = Grad_Plans, y = Salary)) +
  geom_jitter(height = 0.2) +
  labs(title = "Salary vs. Plans")
Salary = subset(Class_Spring_2022_1, Salary < 5000000)
ggplot(data = Salary, aes(x = Grad_Plans, y = Salary )) +
  geom_jitter(height = 0.2) +
  labs(title = "Salary vs. Plans")
Grads = subset(Class_Spring_2022_1, Grad_Plans != "NA" & Salary < 5000000)
ggplot(data = Grads, aes(x = Grad_Plans, y = Salary, fill = Grad_Plans)) + 
  geom_boxplot(color = "black") +
  stat_boxplot(geom = "errorbar") +
  labs(title = "Plan vs. Salary" )
ggplot(data = Class_Spring_2022_1, aes(x = Bones, fill = Academ_Level)) + 
  geom_bar(color = "blue") +
  labs(title = "Bones Broken by Grade")
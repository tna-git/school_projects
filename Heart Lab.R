library(tidyverse)
heart <- read.csv("heart.csv")
view(heart)

ggplot(data = heart, aes(x = fbs, fill = target)) + 
  geom_bar(position = "fill", color = "black", width = 0.4)+
  labs(title = "Fasting Blood sugar and heart disease", x = "Blood sugar above 120", y = "Heart disease proportion") +
  theme_bw() +
  scale_y_continuous(breaks = seq(0,1,0.1))

heart %>%
  group_by(fbs, target) %>%
  summarise(count = n())

prop.test(x = c(116, 22), n = c(258, 45),
          alternative = "greater")

ggplot(data = heart, aes(x = exang, fill = target)) + 
  geom_bar(position = "fill", color = "black", width = 0.4)+
  labs(title = "Exercised induced Angina and heart disease", x = "Patients who had Exercised Induced Angina", y = "Heart disease proportion") +
  theme_bw() +
  scale_y_continuous(breaks = seq(0,1,0.1))

heart %>%
  group_by(exang, target) %>%
  summarise(count = n())

prop.test(x = c(76, 62), n = c(99, 204),
          alternative = "greater")

ggplot(data = heart, aes(x = exang, y = chol, color = exang)) + 
  geom_jitter(width = 0.05)+
  labs(title = "Exercised induced Angina and cholestrol levels", x = "Patients Who Had Exercised Induced Angina", y = "Cholesterol Levels") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_continuous(breaks = seq(126,564,40))

ggplot(data = heart, aes(x = exang, y = chol, group = chol, color = exang)) +
  geom_point(size = 3) +
  geom_line(size = 0.5, color = "black") +
  theme_bw()

t.test(data = heart, chol ~ exang, 
       alternative = "two.sided",
       var.equal = FALSE)

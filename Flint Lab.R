library(tidyverse)
View(Flint)
ggplot(data = Flint , aes(x = Lead_ppb)) +
  geom_density(fill = "green", color = "black") +
  labs(title = "Lead Distribution", x =  "Lead in Parts per Billion(ppb)" +
  scale_x_continuous(breaks = seq(0,1051,100))
  
summary(Flint$Lead_ppb)
ggplot(data = Flint, aes(x = Time, y = Lead_ppb, color = Time)) +
  geom_jitter(width = 0.1, alpha = 0.8) +
  scale_color_brewer(palette = "Set2") +
  labs(title = "Lead_ppb at Different Times", x = "Lead in Parts per Billion(ppb)")

Flint %>%
  group_by(Time) %>%
  summarise(Average = mean(Lead_ppb),
            Median = median(Lead_ppb),
            St_Dev = sd(Lead_ppb))

ggplot(data = Flint, aes(x = Time,  y = Lead_ppb, fill = Time)) + 
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  stat_boxplot(geom = "errorbar") +
  labs(title = "Lead_ppb at Different Times", y = "Lead in Parts per Billion(ppb)") +
  scale_y_continuous(breaks = seq(0,50, 5), limits = c(0, 50)) +
  theme_dark() +
  theme(legend.position = "none")

Flint %>%
  filter(Time == "First Draw") %>%
    group_by(Ward)%>%
      summarise(Average = mean(Lead_ppb), Median = median(Lead_ppb))

Flint_First = subset(Flint, Time == "First Draw")
ggplot(data = Flint, aes(x = Ward , y = Lead_ppb, fill = Time)) + 
  geom_boxplot() +
  stat_boxplot(geom = "errorbar") +
  scale_fill_brewer(palette = "YlGn") +
  labs(title = "Lead_ppb at Different Wards", y = "Lead in Parts per Billion(ppb)") +
  scale_y_continuous(breaks = seq(0,158,5), limits = c(0,50)) +
  theme_dark()

ggplot(data = Flint_First, aes(x = Ward, y = Lead_ppb)) +
  geom_boxplot()

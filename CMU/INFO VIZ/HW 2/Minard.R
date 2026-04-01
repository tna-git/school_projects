library(ggplot2)
library(maps)
library(grid)
library(gridExtra)

# Read in the data files
troops <- read.table("C:/Users/Trish/Documents/CMU/INFO VIZ/HW 2/troops.txt", header = TRUE)
cities <- read.table("C:/Users/Trish/Documents/CMU/INFO VIZ/HW 2/cities.txt", header = TRUE)
temps  <- read.table("C:/Users/Trish/Documents/CMU/INFO VIZ/HW 2/temps.txt",  header = TRUE)
temps$date <- as.Date(strptime(temps$date,"%d%b%Y"))
temps$date_label <- format(temps$date, "%d%b%Y")



# Set exact same x-axis limits for both plots
xlim <- c(24, 38)

# March plot
p_march <- ggplot() +
  geom_path(
    data = troops,
    aes(x = long, y = lat,
        size = survivors,
        colour = direction,
        group = group),
    lineend = "round"
  ) +
  geom_point(data = cities, aes(x = long, y = lat), size = 1.5) +
  geom_text(
    data = cities,
    aes(x = long, y = lat, label = city),
    hjust = 0, vjust = 1, size = 3
  ) +
  scale_size_continuous(
    range  = c(0.5, 12),
    breaks = c(10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000),
    name = "survivors"
  ) +
  scale_colour_manual(
    values = c(A = "#E8746A", R = "#6FCBDB"),
    labels = c(A = "Advancing", R = "Retreating"),
    name = "direction"
  ) +
  scale_x_continuous(limits = xlim, expand = c(0, 0)) +
  scale_y_continuous(limits = c(53.5, 56.5), expand = c(0, 0)) +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing.y = unit(0.3, "cm"),
    plot.margin = unit(c(1, 1, 0.5, 1), "cm"),
    panel.grid.major = element_line(color = "black", size = 0.2),
    panel.grid.minor = element_blank()
  )

# Temperature plot
p_temps <- ggplot(temps, aes(x = long, y = temp)) +
  geom_line(size = 1) +
  geom_text(aes(label = date_label), vjust = -0.5, size = 2.5) +
  scale_x_continuous(limits = xlim, expand = c(0, 0)) +
  scale_y_continuous(limits = c(-32, 5), expand = c(0, 0)) +
  labs(x = NULL, y = "temp") +
  theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    plot.margin = unit(c(0.5, 1, 1, 1), "cm")
  )

# Display
p_march
p_temps

# Combine both plots
p_march <- p_march +
  theme(
    plot.margin = unit(c(1, 1, 1.5, 1), "cm")  # top, right, bottom, left
  )

p_temps <- p_temps +
  theme(
    plot.margin = unit(c(1.5, 1, 1, 1), "cm")
  )


g <- arrangeGrob(p_march, p_temps, ncol = 1, heights = c(11.5, 5.5))

grid.newpage()
grid.draw(g)

ggsave("C:/Users/Trish/Documents/CMU/INFO VIZ/HW 2/minard_aligned.jpg",g, width = 20, height = 20)

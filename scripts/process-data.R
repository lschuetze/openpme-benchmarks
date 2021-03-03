library(dplyr)
library(ggplot2)
library(here)
sourceDir <- getSrcDirectory(function(dummy) {dummy})

row_names <- c("Invocation", "Iteration", "Value", "Unit", "Criterion",
               "Benchmark", "VM", "Approach", "Extra", "Cores", "InputSize",
               "Var")
openfpm_all <- read.table(here(sourceDir, "/../data/benchmark-openfpm.data"), header = FALSE,
                           sep = "\t", col.names = row_names, fill = TRUE)
openpme_LJ <- read.table(here(sourceDir, "../data/benchmark-openpme-LJ.data"), header = FALSE,
                           sep = "\t", col.names = row_names, fill = TRUE)
openpme_LJ_VL <- transform(openpme_LJ, Benchmark = "LennardJonesVL")
openpme_GS <- read.table(here(sourceDir, "../data/benchmark-openpme-GS.data"), header = FALSE,
                         sep = "\t", col.names = row_names, fill = TRUE)
openpme_VIC12 <- read.table(here(sourceDir, "../data/benchmark-openpme-VIC12.data"), header = FALSE,
                         sep = "\t", col.names = row_names, fill = TRUE)
openpme_VIC12_OPT <- transform(openpme_VIC12_OPT, Benchmark = "VortexInCellOpt")

data <- rbind(openfpm_all, openpme_LJ, openpme_LJ_VL,
              openpme_GS, openpme_LJ_VL, openpme_VIC12, openpme_VIC12_OPT)

data <- data %>% filter(Unit == "ms")
data <- transform(data, Value = Value / 1000)
data <- data %>%
  group_by(VM, Benchmark, Approach, Cores) %>%
  summarise(
    Time = mean(Value)
  )


p <- ggplot(data, aes(x = Cores, y = Time, fill = Approach)) +
  labs(y = "Runtime in s") +
  facet_wrap(facets = ~ Benchmark,
             scales = "free", ncol = 3) +
  scale_x_continuous(trans = "log2") +
  geom_bar(stat = "identity", position = "dodge")
print(p)

library(ggplot2)
library(reshape)

df1 <- read.csv("data/B1_loadings.csv")
df2 <- read.csv("data/B2_loadings.csv")

rownames(df1) <- df1$Variable
rownames(df2) <- df2$Variable

# Plot the loadings from a single PCA model
title1 <- "Loadings for PC1-PC2 - PWPCA - B1"
title2 <- "Loadings for PC1-PC2 - PWPCA - B2"

plt_loadings <- function(df, title) {
    load_vars <- df$Variable

    keeps <- c("PC1", "PC2")

    df <- df[, keeps]
    loadings <- melt(
        data.frame(load_vars, df),
        id.vars = c("load_vars"),
        variable.name = "loading"
    )
    colnames(loadings) <- c("Variable", "PC", "Loading")
    dev.new()
    loadings_plot <- ggplot(
            loadings,
            aes(Loading, Variable)
        ) +
        geom_bar(
            stat = "identity",
            fill = "#4682B4"
        ) +
        xlab("Loading value") +
        ylab("Variable") +
        theme_bw() +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1)
        ) +
        facet_wrap(~PC, nrow = 1) +
        ggtitle(title)
    plot(loadings_plot)
    ggsave(
        paste("results/", title, ".pdf", sep = ""),
        plot = loadings_plot,
        device = "pdf"
    )
}

# plt_loadings(df1, title1)
# plt_loadings(df2, title2)

pc1_vs_pc2 <- function(pc_plot, model, save=FALSE) {
    # Scatterplot of PC1 vs PC2 for various models
    dev.new()
    pc_plot <- pc_plot +
        geom_point() +
        theme_bw() +
        #   ylim(c(-300, 100)) +
        #   xlim(c(-5000, 0)) +
        stat_density2d(aes(fill = ..level..), geom = "polygon", alpha = 0.1) +
        xlab("PC1") +
        ylab("PC2") +
        ggtitle(paste("PC1 vs PC2 - ", model)) +
        theme(
            plot.title = element_text(
                color = "black",
                face = "bold",
                size = 14
            )
        ) +
        theme(axis.title = element_text(color = "black", size = 10))
    plot(pc_plot)
    if (save == TRUE) {
        ggsave(
            paste("results/PC1 vs PC2 - ", model, ".pdf", sep = ""),
            plot = pc_plot,
            device = "pdf"
        )
    }
}

scores <- read.csv("data/bib_decorrelated_latent_space.csv")

pwpca_plot <- ggplot(scores, aes(PWPCA_PC1, PWPCA_PC2))

pc1_vs_pc2(pwpca_plot, "PWPCA", save = TRUE)
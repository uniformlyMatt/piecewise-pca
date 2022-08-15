library(corrplot)
library(ggplot2)

plot_correlations <- function(correlation_matrix) {
    # Plot a correlation matrix

    dev.new()
    cplot <- corrplot.mixed(
        correlation_matrix,
        lower = "circle",
        upper = "number",
        upper.col = "black",
        number.cex = .7,
        order = "hclust"
    )

    cplot <- corrplot(
        correlation_matrix,
        type = "lower",
        order = "hclust"
    )
}

plot_histogram <- function(df) {
    # Returns a histogram object - histograms for each column of df
    data_long <- df %>%         # Apply pivot_longer function
      pivot_longer(colnames(df)) %>%
      as.data.frame()

    p <- ggplot(data_long, aes(x = value)) +
      geom_histogram(aes(y = ..density..)) +
      geom_density(col = "#1b98e0", size = 1) +
      facet_wrap(~ name, scales = "free")
    return(p)
}

plot_loadings <- function(loadings, model, save=FALSE) {
    # Plot the loadings from a single PCA model
    title <- paste("Loadings for PC1-PC5 - ", model, sep = "")

    load_vars <- rownames(loadings)
    loadings <- melt(
        data.frame(load_vars, loadings),
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
    if (save == TRUE) {
        ggsave(
            paste("results/", title, ".pdf", sep = ""),
            plot = loadings_plot,
            device = "pdf"
        )
    }
}

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
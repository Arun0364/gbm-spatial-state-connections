suppressMessages(library(ape))
args <- commandArgs(trailingOnly = TRUE)
for (s in c("G4_ST2", "G740_ST1")) {
  C <- as.matrix(read.csv(file.path(args[1], paste0(s, "_centroids.csv")), row.names = 1, check.names = FALSE))
  f <- function(x) {
    z <- scale(x)
    z <- z[, apply(z, 2, function(v) all(is.finite(v))), drop = FALSE]
    nj(dist(z))
  }
  tr <- f(C)
  set.seed(1)
  bp <- boot.phylo(tr, C, f, B = 1000, quiet = TRUE, trees = FALSE)
  parts <- prop.part(tr)
  sp <- lapply(parts, function(p) sort(attr(parts, "labels")[p]))
  cat(s, "ape tree:", write.tree(tr), "\n")
  for (i in seq_along(sp)) cat("  clade", paste(sp[[i]], collapse = ","), "support", bp[i] / 1000, "\n")
}

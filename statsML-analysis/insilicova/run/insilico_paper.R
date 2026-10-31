rm(list = ls())
options(java.parameters = c('-Xmx2g'))

library(InSilicoVAx)
library(tidyverse)
library(jsonlite)

source("../src/clean_champs.R")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   <- "../../data/feb2026/processed"
# Predictions go next to the ML predictions, in the same record format as
# output/paper/ml/infant_child_{lgbm,enet}_pred.jsonl (lower-cased labels, top_1/top_3/all, auxiliary_labels).
JSONL_DIR    <- "../../output/paper/ml"
OUTPUT_JSONL <- file.path(JSONL_DIR, "infant_child_insilicova_pred.jsonl")
dir.create(JSONL_DIR, recursive = TRUE, showWarnings = FALSE)
AUX_COLS <- c("IC_champs_group_desc", sprintf("Morbid_Cond_%02d_champs_group_desc", 1:8))


# Feature settings — mirrors insilico1.R (VA only) and insilico2.R (VA + demo)
FEATURE_SETTINGS <- list(
  list(name = "structured-va",               demo = FALSE),
  list(name = "demographics, structured-va", demo = TRUE)
)

# ---------------------------------------------------------------------------
# Load data — same split as old scripts
# ---------------------------------------------------------------------------

df_train_raw <- read.csv(file.path(DATA_DIR, "champs_infant_child_train.csv"),
                         stringsAsFactors = FALSE)
df_test_raw  <- read.csv(file.path(DATA_DIR, "champs_infant_child_test.csv"),
                         stringsAsFactors = FALSE)

cat(sprintf("Train rows: %d  |  Test rows: %d\n", nrow(df_train_raw), nrow(df_test_raw)))
# Paper cohort check (data/infant_child_split.json): 1939 train / 252 test, built by: python src/data/CHAMPS_train_split.py --data-dir data/feb2026 --split-json data/infant_child_split.json --output-dir data/feb2026/processed
stopifnot(nrow(df_train_raw) == 1939, nrow(df_test_raw) == 252)

# ---------------------------------------------------------------------------
# Cause labels — UC_champs_group_desc (not ICD-11), with injury fix
# ---------------------------------------------------------------------------

cause_train <- df_train_raw[["UC_champs_group_desc"]]
cause_train[which(cause_train == "injury")] <- "Injury"

cause_test <- df_test_raw[["UC_champs_group_desc"]]
cause_test[which(cause_test == "injury")] <- "Injury"
# ML pipeline lower-cases labels; do the same so names match across models
cause_train <- tolower(trimws(cause_train))
cause_test  <- tolower(trimws(cause_test))

stopifnot(length(unique(cause_train)) == 46, length(unique(cause_test)) == 28)
cat(sprintf("Unique train causes: %d  |  Unique test causes: %d\n",
            length(unique(na.omit(cause_train))), length(unique(na.omit(cause_test)))))

# ---------------------------------------------------------------------------
# Helper: run one InSilicoVA setting, return probability matrix
# ---------------------------------------------------------------------------

run_insilico <- function(df_train_raw, df_test_raw, cause_train, demo) {

  df_train <- clean_champs_va_infant(df_train_raw, highSignal = TRUE, demo = demo)
  df_test  <- clean_champs_va_infant(df_test_raw,  highSignal = TRUE, demo = demo)

  prop_missing <- apply(df_train, 2, function(x) mean(x == "."))
  cols_to_keep <- names(prop_missing[prop_missing < 0.9])
  df_train <- df_train[, cols_to_keep]
  df_test  <- df_test[,  cols_to_keep]

  n_test  <- nrow(df_test)
  n_train <- nrow(df_train)

  df_train <- cbind(ID = paste0("d", n_test + seq_len(n_train)), cause = cause_train, df_train)
  df_test  <- cbind(ID = paste0("d", seq_len(n_test)), df_test)

  res <- insilico2(data = df_test, train = df_train, cause = "cause",
                   Nsim = 4000, burnin = 2000)

  as.matrix(get.indiv(res)$median)
}

# ---------------------------------------------------------------------------
# Run all settings and write JSONL predictions
# ---------------------------------------------------------------------------

case_ids <- df_test_raw[["champs_deid"]]
n_test   <- nrow(df_test_raw)

con <- file(OUTPUT_JSONL, open = "w")

for (setting in FEATURE_SETTINGS) {

  cat(sprintf("\n=== Feature setting: %s ===\n", setting$name))

  prob_mat    <- run_insilico(df_train_raw, df_test_raw, cause_train, demo = setting$demo)
  cause_names <- tolower(colnames(prob_mat))   # UC_champs_group_desc labels
  stopifnot(length(cause_names) == 46, !anyDuplicated(cause_names), setequal(cause_names, unique(cause_train)), nrow(prob_mat) == nrow(df_test_raw))

  cat(sprintf("Probability matrix: %d cases x %d causes\n", nrow(prob_mat), ncol(prob_mat)))

  for (i in seq_len(n_test)) {
    probs      <- prob_mat[i, ]
    sorted_idx <- order(probs, decreasing = TRUE)
    sorted_nm  <- cause_names[sorted_idx]

    aux <- list()
    for (col in AUX_COLS) {
      v <- df_test_raw[[col]][i]
      if (!is.na(v) && nzchar(trimws(v)) && !(tolower(v) %in% c("nan", "none", "null")))
        aux[[length(aux) + 1]] <- list(type = col, name = tolower(trimws(v)))
    }

    rec <- list(
      feature_setting  = setting$name,
      champs_deid      = case_ids[i],
      true_label       = list(type = "UC_champs_group_desc", name = cause_test[i]),
      auxiliary_labels = aux,
      predicted_labels = list(
        top_1 = sorted_nm[1],
        top_3 = as.list(sorted_nm[1:3]),
        all   = lapply(sorted_idx, function(j)
                  list(name = cause_names[j], probability = round(probs[j], 4)))
      )
    )
    writeLines(toJSON(rec, auto_unbox = TRUE, digits = NA), con)
  }

  cat(sprintf("Wrote %d records for setting '%s'\n", n_test, setting$name))
}

close(con)
cat(sprintf("\nAll predictions saved to %s\n", OUTPUT_JSONL))

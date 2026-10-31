## clean champs infant module, select verbal autopsy columns

clean_champs_va_infant <- function(df, highSignal = FALSE, demo = TRUE, va = TRUE){

  cols_measure <- c("weight_kg", "height_cm", "MUAC_cm", "head_circumf_cm",
                    "right_leg_length_cm", "right_foot_length_cm")

  df1 <- df[, cols_measure]

  # sex
  df1$male   <- ifelse(is.na(df$sex), ".", ifelse(df$sex == "Male", "Y", ""))
  df1$female <- ifelse(is.na(df$sex), ".", ifelse(df$sex == "Female", "Y", ""))

  # age_group
  df1$child   <- ifelse(is.na(df$age_group), ".", ifelse(df$age_group == "Child (12 months to less than 60 Months)", "Y", ""))
  df1$infant <- ifelse(is.na(df$age_group), ".", ifelse(df$age_group == "Infant (28 days to less than 12 months)", "Y", ""))

  # cols_measure
  for (nm in cols_measure) {
    x <- df1[[nm]]

    q25 <- quantile(x, 0.25, na.rm = TRUE)
    q75 <- quantile(x, 0.75, na.rm = TRUE)

    df1[[paste0(nm, "_lo")]]  <- ifelse(is.na(x), ".", ifelse(x < q25, "Y", ""))
    df1[[paste0(nm, "_med")]] <- ifelse(is.na(x), ".", ifelse(x >= q25 & x <= q75, "Y", ""))
    df1[[paste0(nm, "_hi")]]  <- ifelse(is.na(x), ".", ifelse(x > q75, "Y", ""))
  }
  df1[cols_measure] <- NULL

  # hiv/aids
  hiv_col <- tolower(df[, "Id10002"])
  df1$Id10002_vlo <- ifelse(is.na(hiv_col), ".", ifelse(hiv_col == "veryl", "Y", ""))
  df1$Id10002_lo <- ifelse(is.na(hiv_col), ".", ifelse(hiv_col == "low", "Y", ""))
  df1$Id10002_hi <- ifelse(is.na(hiv_col), ".", ifelse(hiv_col == "high", "Y", ""))
  df1$Id10002_vhi <- ifelse(is.na(hiv_col), ".", ifelse(hiv_col == "very", "Y", ""))

  # malaria
  mal_col <- tolower(df[, "Id10003"])
  df1$Id10003_vlo <- ifelse(is.na(mal_col), ".", ifelse(mal_col == "veryl", "Y", ""))
  df1$Id10003_lo <- ifelse(is.na(mal_col), ".", ifelse(mal_col == "low", "Y", ""))
  df1$Id10003_hi <- ifelse(is.na(mal_col), ".", ifelse(mal_col == "high", "Y", ""))
  df1$Id10003_vhi <- ifelse(is.na(mal_col), ".", ifelse(mal_col == "very", "Y", ""))

  # season
  season_col <- tolower(df[, "Id10004"])
  df1$Id10004_dry <- ifelse(is.na(season_col) | season_col == "dk", ".",
                            ifelse(season_col == "dry", "Y", ""))
  df1$Id10004_wet <- ifelse(is.na(season_col) | season_col == "dk", ".",
                            ifelse(season_col == "wet", "Y", ""))

  tmp <- clean_block(df, "Id10077", "Id10481", recode_yn)
  df1 <- cbind(df1, tmp)

  df1[] <- lapply(df1, function(x) {
    x_chr <- as.character(x)        # ensure character
    factor(x_chr, levels = c("", ".", "Y"))
  })

  # ----------------------------------------
  # IDs to keep
  # ----------------------------------------
  target_ids <- c(
    "Id10077", "Id10079", "Id10084", "Id10085", "Id10089", "Id10090",
    "Id10125", "Id10126", "Id10127", "Id10128", "Id10129", "Id10130",
    "Id10131", "Id10147", "Id10148", "Id10149", "Id10153", "Id10159",
    "Id10166", "Id10181", "Id10186", "Id10188", "Id10191", "Id10193",
    "Id10194", "Id10214", "Id10215", "Id10219", "Id10220", "Id10233",
    "Id10241", "Id10265", "Id10432", "Id10433", "Id10418", "Id10423",
    "Id10420", "Id10421", "Id10462", "Id10464", "Id10466", "Id10472"
  )

  if(highSignal){
    df1_filtered <- keep_matching_columns(df1, target_ids)
    if(demo){
      result <- df1_filtered
    }else{
      result <- keep_matching_columns(df1_filtered, "Id")
    }
  }else{
    if(demo){
      result <- df1
    }else{
      result <- keep_matching_columns(df1, "Id")
    }
  }

  # When va=FALSE, return only demographic columns (non-Id* columns from df1).
  # We extract from df1 directly rather than from result, because the highSignal
  # filter only keeps Id* target columns — demo columns would be lost otherwise.
  if(!va){
    demo_cols <- grep("^Id", colnames(df1), value = TRUE, invert = TRUE)
    result    <- df1[, demo_cols, drop = FALSE]
  }

  return(result)
}

# ----------------------------------------
# Function to filter dataframe columns
# ----------------------------------------
keep_matching_columns <- function(df, target_ids) {

  df_cols <- colnames(df)

  # Store matched columns
  matched_columns <- character()

  # Track which IDs were not found
  not_found_ids <- character()

  for (id in target_ids) {

    # Match exact OR starts with
    matches <- df_cols[
      df_cols == id | startsWith(df_cols, id)
    ]

    if (length(matches) == 0) {
      not_found_ids <- c(not_found_ids, id)
    } else {
      matched_columns <- c(matched_columns, matches)
    }
  }

  matched_columns <- unique(matched_columns)

  # Raise warning if needed
  if (length(not_found_ids) > 0) {
    warning(
      sprintf(
        "%d IDs were not found in dataframe: %s",
        length(not_found_ids),
        paste(not_found_ids, collapse = ", ")
      )
    )
  }

  df[, matched_columns, drop = FALSE]
}

# Recode function: tolower, yes/no/dk/NA → Y / "" / "."
recode_yn <- function(x) {
  x_chr <- tolower(as.character(x))

  # map dk / NA → "."
  x_chr[is.na(x_chr) | x_chr == "dk"] <- "."

  # map yes / no
  x_chr[x_chr == "yes"] <- "Y"
  x_chr[x_chr == "no"]  <- ""

  valid_values <- c("yes", "no", "dk")
  x_chr[!(x_chr %in% valid_values | x_chr %in% c("Y", ""))] <- "."

  x_chr
}

clean_block <- function(df, start_col, end_col, recode_fun) {

  # 1. column range
  col_range <- which(names(df) == start_col) : which(names(df) == end_col)

  # 2. apply recode function
  tmp_list <- lapply(df[, col_range], recode_fun)

  # # 3. drop columns with > 70% "."
  # prop_missing <- sapply(tmp_list, function(x) mean(x == ".", na.rm = TRUE))
  # cols_to_drop <- names(prop_missing[prop_missing > 0.7])
  # tmp_list <- tmp_list[!(names(tmp_list) %in% cols_to_drop)]

  # 4. list → data.frame
  tmp <- as.data.frame(tmp_list, check.names = FALSE)

  # 5. remove columns with ANY NA
  tmp <- tmp[, !sapply(tmp, function(x) any(is.na(x)))]

  return(tmp)
}

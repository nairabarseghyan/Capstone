library(dplyr)
library(lubridate)

process_sales_data <- function(sales_raw) {
  
  
  # Rename revenue column for easy access
  sales_raw <- rename(sales_raw, Revenue_VAT = Revenue.VAT)
  
  
  # Convert 'sale_date' to date format and sort by this date
  sales_raw$sale_date <- as.Date(sales_raw$sale_date)
  sales_raw <- arrange(sales_raw, sale_date)
  
  # Drop rows where 'Revenue_VAT' is NA
  sales_raw <- drop_na(sales_raw, Revenue_VAT)
  
  # Select relevant columns
  final_df <- select(sales_raw, sale_date, Revenue_VAT, Quantity, product_type, Brand, manufacture_country, Branch, City, Districts, Regions, Product_category, IsHoliday)
  
  return(final_df)
}


add_calendar_features <- function(final_df) {
  
  
  # Create a calendar DataFrame based on sale_date
  calendar_df <- data.frame(sale_date = final_df$sale_date)
  calendar_df$date <- as.Date(calendar_df$sale_date)
  calendar_df$month <- month(calendar_df$date)  # Month column
  calendar_df$year <- year(calendar_df$date)  # Year column
  calendar_df$day <- day(calendar_df$date)  # Day column
  calendar_df$day_of_week <- wday(calendar_df$date) - 1  # Adjusted for R (0 = Sunday, 6 = Saturday)
  calendar_df$day_name <- weekdays(calendar_df$date)  # Name of the day
  calendar_df$quarter <- quarter(calendar_df$date)  # Quarter of the year
  calendar_df$is_weekend <- ifelse(calendar_df$day_of_week >= 5, 1, 0)  # Weekend indicator
  calendar_df$week_of_year <- isoweek(calendar_df$date)  # Week of the year
  calendar_df <- calendar_df %>% distinct(date, .keep_all = TRUE)  # Drop duplicate dates
  
  # Merge the calendar_df back into final_df
  final_df <- merge(final_df, calendar_df, by = "sale_date", all.x = TRUE)
  
  # Drop the 'date' column as it's redundant with 'sale_date'
  final_df <- select(final_df, -date)
  
  return(final_df)
}

add_sales_proxies <- function(data) {
  # Make a copy of the data to avoid modifying the original
  data_copy <- data
  
  # Calculate weekly average
  week_average <- data_copy %>%
    group_by(Branch, week_of_year) %>%
    summarise(sales_proxy_week = mean(Revenue_VAT, na.rm = TRUE), .groups = 'drop')
  
  # Merge weekly average
  data_copy <- left_join(data_copy, week_average, by = c("Branch", "week_of_year"))
  
  # Calculate monthly average
  month_average <- data_copy %>%
    group_by(Branch, month) %>%
    summarise(sales_proxy_month = mean(Revenue_VAT, na.rm = TRUE), .groups = 'drop')
  
  # Merge monthly average
  data_copy <- left_join(data_copy, month_average, by = c("Branch", "month"))
  
  # Replace NA with 0 (if any)
  data_copy[is.na(data_copy)] <- 0
  
  return(data_copy)
}


from twitter_scraper_selenium import scrape_profile

# Replace 'elonmusk' with the username you want
# output_format="csv" saves it directly to csv
# browser="chrome" (or "firefox") requires that browser installed on your PC
scrape_profile(
    twitter_username="lonextrades",
    output_format="csv",
    browser="chrome",
    tweets_count=100000,
    directory="."
)
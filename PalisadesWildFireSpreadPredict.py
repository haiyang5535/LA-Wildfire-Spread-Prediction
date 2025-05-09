import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time

# Path to the chromedriver
CHROMEDRIVER_PATH = './chromedriver-mac-arm64/chromedriver'

# Set custom headers to simulate a real browser
caps = DesiredCapabilities().CHROME
caps['goog:loggingPrefs'] = {'performance': 'ALL'}

options = Options()
options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36')
options.add_argument('--disable-gpu')

# Merge capabilities into options
options.add_experimental_option('excludeSwitches', ['enable-automation'])
options.add_experimental_option('prefs', {
    'profile.managed_default_content_settings.images': 2  # Disable images for faster loading
})

service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# Restore the original URL from the CSV file
csv_file = 'calfire_updates_links.csv'
output_csv_file = 'calfire_updates_raw_data.csv'

# Open the output CSV file to save raw data
with open(csv_file, 'r') as file, open(output_csv_file, 'w', newline='') as output_file:
    reader = csv.reader(file)
    writer = csv.writer(output_file)

    # Write header for the output CSV
    writer.writerow(['URL', 'Page Title', 'Page Content'])

    header = next(reader)  # Skip the header row

    for row in reader:
        link = row[0].strip().strip("'").strip('"')  # Get and clean the link

        try:
            print(f"Processing URL: {link}")
            driver.get(link)

            # Scrape the page title and content
            page_title = driver.title
            time.sleep(2)  # Wait for the page to load
            content = driver.find_element(By.TAG_NAME, 'body').text

            # Write the data to the output CSV
            writer.writerow([link, page_title, content])

        except Exception as e:
            print(f"Error processing {link}: {e}")

print(f"Raw data has been saved to {output_csv_file}")

# Close the browser
driver.quit()
from configparser import ConfigParser

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class StartDriver:
    """
    An object that represents a Chrome browser.
    """

    def __init__(self):
        """
        Establishing the path of browser and adding extensions
        """
        config = ConfigParser()
        config.read('sudoku/config.ini')

        chrome_path = config['chrome_info']['driver']
        ad_ex = config['chrome_info']['ad_blocker']

        # Adding ad blocker and starting up browser
        chrome_options = Options()
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.add_argument('load-extension=' + ad_ex)
        self.driver = webdriver.Chrome(chrome_path, options=chrome_options)

    def desired_site(self, site_url):
        """
        Takes the object to a new/another website.

        :param site_url: website user wants to visit
        :type site_url: str
        :return: updated web location
        """
        self.driver.get(site_url)
        return self

    def quit_chrome(self):
        """
        Exits the web browser.

        :return: None
        """
        self.driver.quit()

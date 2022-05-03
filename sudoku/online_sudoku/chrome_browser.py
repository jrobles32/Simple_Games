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
        chrome_path = 'D:/Py_ChromeDriver/chromedriver.exe'
        ad_ex = 'C:/Users/Javier/AppData/Local/Google/Chrome/User ' \
                'Data/Default/Extensions/cjpalhdlnbpafiamejdnhcphjbkeiagm/1.42.4_0 '

        # Adding ad blocker and starting up browser
        chrome_options = Options()
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

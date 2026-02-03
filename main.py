# -*- coding: utf-8 -*-
from search_bot.app import BotApp

if __name__ == "__main__":
    try:
        BotApp().run()
    except KeyboardInterrupt:
        import logging
        logging.getLogger(__name__).info("Stopped by user")

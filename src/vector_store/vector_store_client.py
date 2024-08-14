# pylint:disable = import-error, too-few-public-methods, broad-exception-caught
"""
This module provides a client for connecting to a Weaviate server.
"""
import time
import weaviate
from loguru import logger

class WeaviateClient:
    """
    A client class for establishing a connection to a Weaviate server.
    """

    def __init__(self, host, port):
        """
        Initializes the Weaviate client. This method continuously checks if the Weaviate server is
        available by calling 'check_weaviate_server' until a successful connection is established.
        :param: host (str): The hostname or IP address of the Weaviate server.
        :param: port (int): The port number on which the Weaviate server is running.
        """
        while not self.check_weaviate_server(host, port):
            logger.debug("Waiting for the Weaviate server to start...")
            time.sleep(5)
        logger.info("Weaviate Server is running!")

        self.client = weaviate.connect_to_local(host=host, port=port)

    @staticmethod
    def check_weaviate_server(
            host: str,
            port: int
    ) -> bool:
        """
        Checks if the Weaviate server is available. This static method tries to connect
        to the Weaviate server using the provided host and port. It returns 'True' if
        the connection is successful, otherwise 'False'.
        :param: host (str): The hostname or IP address of the Weaviate server.
        :param: port (int): The port number on which the Weaviate server is running.
        :return bool: 'True' if the server is reachable, 'False' otherwise.
        """
        try:
            weaviate.connect_to_local(host=host, port=port)
            return True
        except Exception:
            return False

from abc import ABC, abstractmethod


class NotificatorInterface(ABC):

    @abstractmethod
    def send_notification(self, message: str):
        pass

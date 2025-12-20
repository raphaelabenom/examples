from .notificator_interface import NotificatorInterface


class NotificatorEmail(NotificatorInterface):
    def send_notification(self, message: str):
        print(f"Sending Email: {message}")

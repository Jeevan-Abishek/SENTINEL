from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from loguru import logger


class NotificationChannel(ABC):
    @abstractmethod
    def send(self, alert: "Alert", **kwargs) -> bool:
        pass


class WebSocketNotificationChannel(NotificationChannel):
    def __init__(self, ws_manager):
        self.ws_manager = ws_manager

    async def send(self, alert: "Alert", **kwargs) -> bool:
        try:
            message = {
                "type": "alert",
                "data": {
                    "id": alert.id,
                    "zone_id": alert.zone_id,
                    "risk_level": alert.risk_level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                }
            }
            await self.ws_manager.broadcast(message)
            return True
        except Exception as e:
            logger.error(f"WebSocket notification failed: {e}")
            return False


class EmailNotificationChannel(NotificationChannel):
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config

    async def send(self, alert: "Alert", **kwargs) -> bool:
        # Placeholder for email sending
        logger.info(f"Email alert sent for {alert.id}")
        return True


class SMSNotificationChannel(NotificationChannel):
    def __init__(self, provider_config: Dict[str, Any]):
        self.provider_config = provider_config

    async def send(self, alert: "Alert", **kwargs) -> bool:
        # Placeholder for SMS sending
        logger.info(f"SMS alert sent for {alert.id}")
        return True


class SoundNotificationChannel(NotificationChannel):
    def __init__(self, sound_files: Dict[str, str]):
        self.sound_files = sound_files

    def send(self, alert: "Alert", **kwargs) -> bool:
        # Placeholder for sound playback
        sound_file = self.sound_files.get(alert.risk_level.value, "default.wav")
        logger.info(f"Playing alert sound: {sound_file} for {alert.id}")
        return True


class NotificationService:
    def __init__(self):
        self.channels: list[NotificationChannel] = []

    def add_channel(self, channel: NotificationChannel):
        self.channels.append(channel)

    async def notify(self, alert: "Alert"):
        for channel in self.channels:
            try:
                if hasattr(channel, 'send') and callable(channel.send):
                    import asyncio
                    if asyncio.iscoroutinefunction(channel.send):
                        await channel.send(alert)
                    else:
                        channel.send(alert)
            except Exception as e:
                logger.error(f"Notification channel error: {e}")
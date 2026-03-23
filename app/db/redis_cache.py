import json
import redis.asyncio as redis
from app.core.config import settings

class RedisCache:
    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        
    async def get(self, key: str):
        val = await self.client.get(key)
        if val:
            return json.loads(val)
        return None
        
    async def set(self, key: str, value: dict, expire: int = 3600):
        await self.client.set(key, json.dumps(value), ex=expire)
        
redis_client = RedisCache()
